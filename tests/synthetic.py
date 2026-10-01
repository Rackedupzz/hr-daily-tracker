"""A small synthetic season in the PA feed's format, for tests that need a whole
pipeline to run: ten clubs in their own parks, forty days, switch hitters,
bench players who pinch-hit, starters and relievers, weather that varies."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HIT_TYPES = ("single", "double", "triple")


def write_season(path: Path, days: int = 40, clubs: int = 10, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    teams = [f"Club {i}" for i in range(clubs)]
    park = {t: f"Park {i}" for i, t in enumerate(teams)}
    park_hr = {t: float(rng.uniform(0.8, 1.25)) for t in teams}
    hitters, power, bats = {}, {}, {}
    for i, t in enumerate(teams):
        ids = [1000 + 20 * i + j for j in range(12)]
        hitters[t] = ids
        for j, b in enumerate(ids):
            power[b] = float(rng.uniform(0.012, 0.065))
            bats[b] = "S" if j == 0 else ("L" if rng.random() < 0.4 else "R")
    staff = {t: [5000 + 20 * i + j for j in range(9)] for i, t in enumerate(teams)}
    throws = {p: ("L" if rng.random() < 0.3 else "R") for t in teams for p in staff[t]}
    allow = {p: float(rng.uniform(0.75, 1.3)) for t in teams for p in staff[t]}

    game_pk, lines, rotation = 700000, [], {t: 0 for t in teams}
    start = date(2026, 4, 1)
    for d in range(days):
        day = str(start + timedelta(days=d))
        order = teams[d % clubs:] + teams[:d % clubs]
        for g in range(clubs // 2):
            home, away = (order[g], order[-1 - g]) if d % 2 else (order[-1 - g], order[g])
            game_pk += 1
            temp = float(rng.choice([52.0, 64.0, 76.0, 88.0]))
            wind = str(rng.choice(["Out To CF", "In From LF", "L To R", "Calm"]))
            mph = float(rng.integers(0, 16))
            pa_index = 0
            lineups, starters = {}, {}
            for t in (home, away):
                lineups[t] = list(rng.permutation(hitters[t][:11])[:9])
                starters[t] = staff[t][rotation[t] % 5]
                rotation[t] += 1
            spot = {home: 0, away: 0}
            faced = {}
            for inning in range(1, 10):
                for half, (bat, pit) in enumerate(((away, home), (home, away))):
                    outs = 0
                    while outs < 3:
                        slot = spot[bat] % 9
                        batter = lineups[bat][slot]
                        if inning >= 8 and slot == 8 and rng.random() < 0.5:
                            batter = hitters[bat][11]           # a pinch hitter in slot 9
                        pitcher = starters[pit] if faced.get(starters[pit], 0) < 22 \
                            else staff[pit][5 + (inning % 4)]
                        faced[pitcher] = faced.get(pitcher, 0) + 1
                        hand = throws[pitcher]
                        side = bats[batter] if bats[batter] != "S" else ("R" if hand == "L" else "L")
                        platoon = 0.75 if side == hand else 1.08
                        p_hr = power[batter] * park_hr[home] * platoon * allow[pitcher] * (
                            1 + 0.01 * (temp - 72))
                        u = rng.random()
                        if u < p_hr:
                            event, ev, la = "home_run", rng.uniform(100, 115), rng.uniform(22, 36)
                        elif u < p_hr + 0.16:
                            event = str(rng.choice(HIT_TYPES, p=[0.75, 0.22, 0.03]))
                            ev, la = rng.uniform(80, 108), rng.uniform(-5, 25)
                        elif u < p_hr + 0.38:
                            event, ev, la = "strikeout", None, None
                        elif u < p_hr + 0.46:
                            event, ev, la = "walk", None, None
                        else:
                            event, ev, la = "field_out", rng.uniform(60, 104), rng.uniform(-30, 60)
                        if event in ("strikeout", "field_out"):
                            outs += 1
                        lines.append(json.dumps({
                            "game_pk": game_pk, "date": day, "venue": park[home],
                            "batting_team": bat, "pitching_team": pit, "is_home": half,
                            "batter_id": int(batter), "batter": f"Hitter {batter}",
                            "bat_side": side, "pitcher_id": int(pitcher),
                            "pitcher": f"Pitcher {pitcher}", "pitch_hand": hand,
                            "inning": inning, "pa_index": pa_index, "event": event,
                            "event_type": event, "is_hr": int(event == "home_run"),
                            "launch_speed": None if ev is None else round(float(ev), 1),
                            "launch_angle": None if la is None else round(float(la), 1),
                            "pull_angle": None if ev is None else round(float(rng.uniform(-40, 40)), 1),
                            "lineup_slot": slot + 1, "tto": faced[pitcher] // 9 + 1,
                            "temp_f": temp, "wind_mph": mph, "wind_dir": wind,
                            "condition": "Clear", "ump_hp": "Ump",
                        }))
                        pa_index += 1
                        spot[bat] += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"teams": teams, "park": park, "hitters": hitters, "bats": bats}
