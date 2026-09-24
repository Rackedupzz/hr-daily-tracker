"""ESPN API client: player bio, physical attributes, and injury status.

Supplies the contextual data the MLB Stats API PA feed does not carry:

- **Bio/physical** (height, weight, age, bat side) -- power proxies used as
  KNN comparables features. Physique and age correlate with raw power without
  being derived from home-run totals, so they add signal without target leakage.
- **Injury status** -- the slate previously could pick a player on the 60-day
  IL. ESPN publishes a live league-wide injury report; anyone on an IL variant
  is dropped from the eligible pool.

All responses are cached to disk so repeated app restarts do not re-hit ESPN.
"""
from __future__ import annotations

import json
import time
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"

# IL designations mean the player cannot appear. "Day-To-Day" players usually
# still play, so they are flagged but remain eligible.
UNAVAILABLE_STATUSES = {
    "60-day-il", "15-day-il", "10-day-il", "7-day il", "7-day-il",
    "out", "suspension",
}

_PITCHER_GROUPS = {"pitchers"}

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
_STRIP_CHARS = ".,'’"


def normalize_name(name: str) -> str:
    """Fold a player name to a match key.

    ESPN and the MLB Stats API disagree on accents, punctuation and suffixes
    (Ali Sanchez vs Ali Sanchez, Ronald Acuna Jr.), so both sides are folded to
    ASCII, lowercased, and stripped of punctuation and generational suffixes.
    """
    if not name:
        return ""
    # Strip accents: decompose, then drop combining marks.
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in decomposed if not unicodedata.combining(c))
    ascii_name = ascii_name.lower().replace("-", " ")
    for ch in _STRIP_CHARS:
        ascii_name = ascii_name.replace(ch, "")
    parts = [p for p in ascii_name.split() if p not in _NAME_SUFFIXES]
    return " ".join(parts)


@dataclass
class PlayerBio:
    """Physical and biographical attributes for one player."""
    espn_id: str
    name: str
    team: str
    height_in: Optional[float]   # inches
    weight_lb: Optional[float]   # pounds
    age: Optional[int]
    bats: str                    # 'L', 'R', 'S' (switch), or '' if unknown
    position: str
    is_pitcher: bool


class ESPNData:
    """Fetches and caches ESPN roster bio + injury data, keyed by folded name."""

    def __init__(self, cache_path: str, ttl_hours: float = 12.0, timeout: int = 20):
        self.cache_path = Path(cache_path)
        self.ttl_seconds = ttl_hours * 3600
        self.timeout = timeout
        self.bios: dict[str, PlayerBio] = {}       # folded name -> bio
        self.injuries: dict[str, str] = {}         # folded name -> status
        self.loaded_from_cache = False

    # ---------------------------------------------------------------- fetching

    def _get(self, url: str, attempts: int = 3) -> Optional[dict]:
        """GET JSON with linear backoff.

        Returns None rather than raising, so an ESPN outage degrades the model
        to its MLB-only features instead of breaking the page.
        """
        for attempt in range(attempts):
            try:
                resp = requests.get(url, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001 - any failure is non-fatal
                if attempt == attempts - 1:
                    print(f"[espn.py] give up on {url}: {exc}")
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    def _cache_is_fresh(self) -> bool:
        if not self.cache_path.exists():
            return False
        age = time.time() - self.cache_path.stat().st_mtime
        return age < self.ttl_seconds

    def load(self, force_refresh: bool = False) -> "ESPNData":
        """Populate from cache when fresh, otherwise from the ESPN API."""
        if not force_refresh and self._cache_is_fresh():
            try:
                blob = json.loads(self.cache_path.read_text(encoding="utf-8"))
                self.bios = {
                    k: PlayerBio(**v) for k, v in blob.get("bios", {}).items()
                }
                self.injuries = blob.get("injuries", {})
                self.loaded_from_cache = True
                print(
                    f"[espn.py] cache hit: {len(self.bios)} bios, "
                    f"{len(self.injuries)} injuries"
                )
                return self
            except Exception as exc:  # noqa: BLE001 - corrupt cache -> refetch
                print(f"[espn.py] cache unreadable ({exc}); refetching")

        self._fetch_rosters()
        self._fetch_injuries()
        self._save_cache()
        return self

    def _fetch_rosters(self) -> None:
        """Pull every team roster for bio/physical attributes."""
        teams_blob = self._get(f"{ESPN_BASE}/teams")
        if not teams_blob:
            print("[espn.py] could not list teams; bio features unavailable")
            return

        try:
            team_entries = teams_blob["sports"][0]["leagues"][0]["teams"]
        except (KeyError, IndexError):
            print("[espn.py] unexpected teams payload; bio features unavailable")
            return

        for entry in team_entries:
            team = entry.get("team", {})
            team_id, team_name = team.get("id"), team.get("displayName", "")
            if not team_id:
                continue
            roster = self._get(f"{ESPN_BASE}/teams/{team_id}/roster")
            if not roster:
                continue
            for group in roster.get("athletes", []):
                group_label = str(group.get("position", "")).lower()
                is_pitcher_group = group_label in _PITCHER_GROUPS
                for athlete in group.get("items", []):
                    self._add_bio(athlete, team_name, is_pitcher_group)

        print(f"[espn.py] loaded {len(self.bios)} player bios from ESPN rosters")

    def _add_bio(self, athlete: dict, team_name: str, is_pitcher_group: bool) -> None:
        name = athlete.get("displayName") or athlete.get("fullName") or ""
        key = normalize_name(name)
        if not key:
            return
        position = (athlete.get("position") or {}).get("abbreviation", "")
        bats = (athlete.get("bats") or {}).get("abbreviation", "") or ""
        self.bios[key] = PlayerBio(
            espn_id=str(athlete.get("id", "")),
            name=name,
            team=team_name,
            height_in=_as_float(athlete.get("height")),
            weight_lb=_as_float(athlete.get("weight")),
            age=_as_int(athlete.get("age")),
            bats=bats.upper()[:1],
            position=position,
            is_pitcher=is_pitcher_group or position in ("SP", "RP", "P"),
        )

    def _fetch_injuries(self) -> None:
        """Pull the league-wide injury report."""
        blob = self._get(f"{ESPN_BASE}/injuries")
        if not blob:
            print("[espn.py] could not load injuries; no availability filtering")
            return
        for team_block in blob.get("injuries", []):
            for item in team_block.get("injuries", []):
                athlete = item.get("athlete") or {}
                key = normalize_name(athlete.get("displayName", ""))
                if key:
                    self.injuries[key] = item.get("status", "")
        print(f"[espn.py] loaded {len(self.injuries)} injury entries")

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            blob = {
                "fetched_at": time.time(),
                "bios": {k: asdict(v) for k, v in self.bios.items()},
                "injuries": self.injuries,
            }
            self.cache_path.write_text(json.dumps(blob), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - cache write is best-effort
            print(f"[espn.py] could not write cache: {exc}")

    # ---------------------------------------------------------------- querying

    def bio_for(self, name: str) -> Optional[PlayerBio]:
        """Bio by player name, tolerant of accent/suffix differences."""
        return self.bios.get(normalize_name(name))

    def injury_status(self, name: str) -> Optional[str]:
        """Injury status string, or None if the player is not on the report."""
        return self.injuries.get(normalize_name(name))

    def is_unavailable(self, name: str) -> bool:
        """True when the player is on an IL variant and cannot play today."""
        status = self.injury_status(name)
        if not status:
            return False
        return status.strip().lower() in UNAVAILABLE_STATUSES

    def coverage(self, names: list[str]) -> dict[str, float]:
        """Diagnostic: what fraction of the given names matched ESPN bios."""
        if not names:
            return {"matched": 0, "total": 0, "rate": 0.0}
        matched = sum(1 for n in names if normalize_name(n) in self.bios)
        return {
            "matched": matched,
            "total": len(names),
            "rate": matched / len(names),
        }


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
