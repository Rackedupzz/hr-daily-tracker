# 🏟️ HR Daily Tracker

A modern, PA-based home run probability model that predicts daily HR slates for MLB games. Features a beautiful web interface with real-time data auto-updates and runs in Docker.

## Features

- **PA-Based Model**: Uses real plate-appearance exposure (not team-games assumptions)
- **Per-Batter Splits**: Tracks HR probability vs left and right-handed pitchers separately
- **Empirical Bayes Shrinkage**: Proper handling of small-sample uncertainty
- **6-Pick Slate Generator**: Daily randomizer produces 6 HR player picks from today's games
- **Beautiful UI**: Modern, responsive design with dark/light mode support
- **Docker Ready**: Run locally in seconds with no dependencies
- **Comparables Model**: KNN finds similar hitters by exit velocity, barrel rate and physique to set each batter's prior
- **ESPN Integration**: Player bio and physical attributes as model features, plus live injury filtering
- **Walk-Forward Validation**: Fitted on early season, scored on held-out later PAs
- **Auto-Update**: Fetches latest 2026 season data on startup
- **Live Results**: Every refresh pulls the day's scores and box scores; picks are
  marked hit or miss as games finish, and each model variant is scored on the same day
- **Model Lab**: A second page comparing every prior side by side on the same slate

## Quick Start (Docker)

### Prerequisites
- Docker and Docker Compose installed

### Run
```bash
docker-compose up
```

Open http://localhost:5000 in your browser.

## Kubernetes (alternative)

```bash
docker build -t hr-daily-tracker:latest .
kubectl apply -k k8s/
kubectl -n hr-daily-tracker port-forward svc/hr-daily-tracker 5000:80
```

Runs as a single replica with persistent volumes for `data/` and `snapshots/`;
an init container seeds the data volume from the image on first boot. Optional
API keys go in a `hr-daily-tracker-keys` Secret (see `k8s/secret.example.yaml`).

## Data on Hugging Face

`src/mlb_hr/data/` (season PAs, fitted models) is gitignored and kept in a
private Hugging Face dataset instead. Set `HF_TOKEN` and `HF_DATA_REPO` in
`.env` (see `.env.example`); the container then downloads any missing data
files on start. After refits, upload the current data with:

```bash
docker compose exec hr-tracker python -m mlb_hr.hf_data push
```

## Manual Setup (VS Code / Local)

### Prerequisites
- Python 3.12+
- pip

### Install
```bash
pip install -r requirements.txt
```

### Run
```bash
python -m mlb_hr.app
```

Open http://localhost:5000 in your browser.

## Project Structure

```
mlb-hr-model/
├── src/mlb_hr/
│   ├── __init__.py          # Package marker
│   ├── fetch.py             # MLB Stats API client
│   ├── model.py             # PA-based HR probability model
│   ├── features.py          # Batter profiles + comparables feature matrix
│   ├── ensemble.py          # 10 priors (KNN/SVR/forest/linear/logistic/XGB/torch) + EB
│   ├── espn.py              # ESPN bio + injury client (disk-cached)
│   ├── slate.py             # Game slate + 6-pick generator
│   ├── results.py           # Live scores + scoring picks against outcomes
│   ├── render.py            # HTML renderers (slate + model lab)
│   ├── render_review.py     # HTML renderers (results review + HOMER)
│   ├── homer.py             # HOMER, the independent expert bot
│   ├── app.py               # Flask server
│   └── data/
│       └── season_pa.jsonl  # Full 2026 season PA data (~165K records)
├── Dockerfile               # Container definition
├── docker-compose.yml       # Compose configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## How It Works

### 1. Data Collection (`fetch.py`)
- Fetches every plate appearance from 2026 season via MLB Stats API
- Resumable JSONL format (165,045 PAs across 2,181 games as of Sep 8)
- Captures real exposure: PA count, games played, handedness splits

### 2. Model Building (`model.py`)
- Aggregates PA data to per-batter, per-pitcher-handedness stats
- Applies empirical-Bayes shrinkage with K=10-50 (tuned per PA volume)
- Fixes the exposure bias from previous team-games-denominator model
- Validates via Brier score, log loss, calibration metrics

### 3. Slate Generation (`slate.py`)
- Fetches today's probable pitchers and active rosters
- Builds slate of eligible hitters (active roster + ≥2 season PAs)
- Computes per-batter HR probability vs each pitcher's handedness
- Generates 6 top picks (one per game where possible, enforcing team diversity)

### 4. Web Rendering (`render.py`)
- Generates responsive HTML with modern styling
- Shows 6 picks with HR probabilities and season stats
- Lists all games with top hitters and probable pitchers
- Analytics dashboard with model stats
- Dark/light mode support

### 5. Server (`app.py`)
- Flask server that auto-fetches data on startup
- Finds latest date with games, renders slate
- Serves on http://localhost:5000
- Health check endpoint at /health
- The slate is built once per date and cached in-process, so both pages render
  from the same slate instead of refitting the ensemble on every request

### 6. Live data & results (`results.py`)

The page has two clocks. The fitted slate is expensive (minutes) and is
refreshed on a timer **in the background**, so a stale model never blocks a
reader. Scores and box scores are two cheap API calls and are pulled on **every
page load**, so a refresh after the last out posts the outcome.

| Layer | Source | Refreshed |
|---|---|---|
| Fitted slate (picks, probabilities) | season PA file + ensemble | every `SLATE_TTL_MINUTES` (default 45), in a background thread |
| Game state and score | `schedule?hydrate=linescore`, one call | every request (`RESULTS_TTL_SECONDS`, default 45) |
| Batting lines (HR, hits, PA) | `game/{pk}/boxscore`, started games only | every request, same TTL |

What shows up once games begin:

- **Results banner** — how many of the six picks connected, out of the games
  that are final, plus how many hit picks got a hit.
- **Per-pick badge** — ✓ with the home run count, ✗ once the game is final with
  none, or a live badge with the line so far. A pick is never a miss while he
  still has at-bats left.
- **Game cards** — live score and inning, or the final.
- **Still to Play** — a section holding just the games that are not final, with
  each one's top home run probabilities and projected hits. Once the afternoon
  is settled, this is the part of the slate that is still actionable; the same
  data is served as JSON at `/api/remaining`.
- **Model Lab** — each prior's own top six carries its own hit tally, so the
  priors are scored against the same day, not just against each other.

Both footers carry two timestamps: when the model was fit and when the live
data was pulled.

## Pages

| Route | What it shows |
|---|---|
| `/` | The daily slate: 6 picks, every game, hits and strikeout projections, and results once games start |
| `/results` | **Results** — a previous day (defaults to yesterday, `?date=YYYY-MM-DD` for any archived day): picks vs box scores, every home run on the slate with its model rank, final scores vs projected winners and totals, and HOMER's card |
| `/homer` | **HOMER** — the resident MLB expert bot: his own 6 HR picks, a winner for every game, his reasoning, where he disagrees with the model, and his season record (`?date=` for past cards) |
| `/models` | **Model Lab** — every prior side by side on the same slate |
| `/api/remaining` | JSON: HR and hit projections for games that are not final |
| `/api/homer` | JSON: HOMER's current card and season record |
| `/health` | JSON health check |

The pages are linked by a tab bar in the header.

### HOMER (`homer.py`)

HOMER is the tracker's top-tier MLB expert and professional capper persona: six
HR picks, six hit picks, a winner for every game, and **HOMER'S PLAYS** — five
parlays that every one had to clear the same bar in the season replay: cash on
at least three days in ten. That bar decides the board. A parlay can never beat
its weakest leg and HOMER's best HR bat cashes ~21%, so no play carries a home
run leg; two legs cannot fall below 46% and five cannot reach 30%, so every play
is three or four hit legs. What separates them is how many legs are taken for
price rather than for likelihood: Parlay of the Day (3 legs, none for value),
The Lock (3, one for value), The Edge (3, two), The Stretch (4, one) and Full
Value (3, all three).

Legs always come from different games; each play shows its combined probability
and fair American odds, and a leg whose player never bats is voided. Every play
type is rebuilt on every day of the season replay, so the page shows how often
it actually cashed against what was predicted. A play's key in the replay record
pins its shape — leg count, HR legs and value legs — so a reshaped play needs a
new key rather than inheriting the old one's track record. His swagger (HAMMER IT / LOVE IT / ...) follows the grade,
and the grade follows the replay.

HOMER is an independent handicapper — none of the ensemble's probabilities feed
him. From the slate he takes only facts (lineups, starters, venue, weather) and
from the season PA feed he builds his own book in one ~30 second pass:

- **Expected HR** from exit velocity × launch angle bins (league HR rate per bin)
- **Form and splits**: last-21-day xHR, his own record vs today's pitcher hand, barrel rate, platoon
- **Pitchers**: HR/xHR allowed, fly-ball rate and recent form for the starter; the bullpen behind him
- **Park** factors from home vs road HR/PA; **weather** rules for temperature and wind to the pull field
- **Lineup**: posted batting order, or his usual recent spot until it posts
- **Games**: wOBA runs model + Pythagenpat, blended with season run differential

**He learns what matters** (`homer_fit.py`). The season is replayed week by
week with his book frozen at each week's start; weights are fitted on earlier
weeks and graded on the next. Factors are dropped while dropping one improves
held-out log loss, learned weights are served only if they beat the
rule-of-thumb formula, probabilities are recalibrated to the replay's actual
hit rates, and letter grades are bands that each homered more often than the
band below. Hit picks (1+ hit) are learned and tested the same way. The HOMER
tab's **Homework** section shows all of it. The app re-runs the replay daily
(when no slate fit is in flight), so every day's results reshape him; by hand:

```bash
docker compose exec hr-tracker python -m mlb_hr.homer_fit
```

Replay through 2026-09-13 (28,773 held-out hitter-games, 121 days):

| | HR picks hit (daily 6) | AUC |
|---|---|---|
| HOMER, learned | 20.0% (144/719) | 0.611 |
| HOMER, rule-of-thumb | 18.8% | 0.605 |
| Season HR rate only | 19.5% | 0.590 |
| Any hitter in the lineup | 12.4% | — |

Winners: 55.3% (house model 54.7%, home team 52.8%); "Strong" calls won 61.8%.

Cards are saved to `snapshots/homer/<date>.json`. Picks in started games are
locked, just like the house slate, and cards are graded automatically as games
finish. Cards built after the fact are marked *backfilled*:

```bash
docker compose exec hr-tracker python -m mlb_hr.homer 2026-09-13
```

### Model Lab (`/models`)

The slate page shows one probability per hitter — whichever prior the ensemble
is configured to serve. The Model Lab shows the same hitters under *all* of
them, so the prior is the only thing that changes across the columns:

| Prior | What it is |
|---|---|
| Flat league | The league HR rate for everyone — the null model |
| KNN comps | Pooled rate of the nearest comparables (served, inside the core blend) |
| SVR | Support-vector regression on the rate |
| Random forest | Forest regression; a learned similarity metric |
| **Linear regression** | sklearn `LinearRegression`, PA-weighted least squares |
| **Logistic regression** | sklearn `LogisticRegression` on the per-PA Bernoulli outcome |
| **Neural net** | PyTorch MLP (2 hidden layers, sigmoid-bounded output) |
| **XGBoost** | Gradient-boosted trees, depth 3 with subsampling and L2 |
| Core / Forest blend | Weighted combinations, weights fitted out-of-fold |

Every prior is fit on the identical feature matrix and cross-validated the same
way, so the comparison isolates the model. The four marked in bold are
**comparison-only** — they are scored but do not feed the served blend, which
stays KNN + SVR until something beats it on held-out data.

Two notes on the newer priors. The logistic model is the only one that models
the actual Bernoulli outcome rather than regressing a rate; it is fit on two
rows per batter (home runs, and the plate appearances that were not) with count
weights, which is the same binomial fit as expanding to 130,000 rows at a
six-hundredth of the size. Least squares is unbounded and will happily predict
a negative home-run rate for a weak hitter, which is why every prior is clipped
to a plausible range before it is used.

The page shows:

- **Held-out scores** — AUC, top-decile lift, Brier, log loss and calibration
  RMSE per prior, from the walk-forward split, with the served prior marked.
- **Thin-sample subset** — the same scores restricted to hitters with thin
  pre-cutoff records, which is the only population where the priors separate.
- **Slate agreement** — each prior's own top six, and how many names it shares
  with the served six. Two priors can score identically per plate appearance
  and still hand you a different slate; this is where that shows up.
- **Hitter by hitter** — a table of per-game probabilities from every prior for
  the top hitters on the slate, served column highlighted.

## Is it actually learning? (`diagnostics.py`)

A held-out AUC of 0.58 on a 3% base rate looks like a broken model and is also
exactly what a working one looks like when the event is mostly noise. Four
tests separate those cases:

```bash
docker compose run --rm hr-tracker python -m mlb_hr.diagnostics
```

| Test | What it proves | Fails if |
|---|---|---|
| **Recovery** | Fits the same estimators on synthetic data with a known strong signal | A model can't recover a signal that is definitely there → the code is broken |
| **Permutation** | Shuffles targets across batters and refits | Any skill remains → leakage |
| **Learning curve** | Refits on 25/50/75/100% of batters, fixed held-out evaluation set | Error doesn't fall as data grows → not learning |
| **Noise ceiling** | Simulates outcomes from known talent and ranks by that same truth | — it's the best score any predictor can achieve |

The evaluation batters in the learning curve are held out of every fit and kept
fixed, so sample size is the only thing that changes. Scoring whoever happens
to be in the subsample instead confounds it: adding thin-record hitters raises
the error regardless of what the model learned, which reads as the model
getting worse when it isn't.

The ceiling matters most. Per-PA prediction cannot approach AUC 1.0 — with a
~3% base rate, even a predictor that *is* the data-generating process scores
about **0.62**. That is the number the model's 0.575 should be read against,
not 1.0.

## Model Details

### Why PA-Based Exposure?
Previous model used `team_games_played × 9 positions` as a hitter's exposure denominator. This assumed every batter started every game—clearly wrong for part-time players and recent acquisitions.

New model uses **actual plate appearances**, fixing this bias:
- **Pete Crow-Armstrong**: Was overstated at 24.7%, now calibrated to 6.2% (matches 41 HR in 655 PA)
- **Junior Caminero**: Was understated at 6%, now shows 6.1% (39 HR in 630 PA)
- **Aggregate median bias**: +2% -> all fixed via handedness splits

(Those percentages are per plate appearance; the UI headline is per game.)

### Shrinkage Formula
```
P(HR per PA) = (observed_hrs + K x prior) / (observed_pas + K)
```
Where:
- `prior` = the batter's **KNN comparables rate** (not a flat league constant)
- `K` = shrinkage strength, estimated by method of moments from the residual
  variance around the prior, not hand-tuned

The prior is the pooled HR rate of the batter's 15 nearest comparables, with
the batter himself excluded from his own neighbour set. A rookie with 30 PA is
pulled toward what similar hitters do given his exit velocity, physique and
age; a regular with 600 PA barely moves.

### Comparables Features
Neighbours are found on quality-of-contact and physical attributes — never on
home-run counts, which are the prediction target:

| Source | Features |
|---|---|
| MLB PA feed (batted ball) | avg exit velocity, 90th-pct EV, hard-hit rate, barrel rate, fly-ball rate, avg launch angle |
| MLB PA feed (approach) | PA volume, balls-in-play rate, K rate, BB rate, non-HR extra-base-hit rate |
| ESPN | height, weight, age |

Features are standardized and then weighted by their correlation with HR rate,
so "nearest" means nearest in the directions that track home-run production.
Barrel rate, hard-hit rate and 90th-percentile exit velocity dominate.

### Per-Game vs Per-PA
The model estimates a **per-PA** rate. The headline number on each pick is the
**per-game** probability, converted through that batter's own plate appearances
per game:

```
P(HR in game) = 1 - (1 - p_per_pa) ^ expected_PA
```

### Platoon Splits
Left/right multipliers are measured from the season's own PA data per
(bat side, pitcher hand) matchup, replacing hardcoded 1.10 / 0.93 constants.

### Park Factors
Home run-friendly parks adjusted upward (Coors +22%), pitcher parks downward (Oracle -22%).

## API Reference

### `HRModel`
Load and query PA-based probabilities:
```python
model = HRModel('src/mlb_hr/data/season_pa.jsonl')
batter = model.get_batter_by_name('Pete Crow-Armstrong')
prob = batter.hr_rate_vs('L')  # HR% vs LHP
```

### `GameSlate`
Build today's slate:
```python
from mlb_hr.slate import build_slate_for_date
from datetime import date

slate = build_slate_for_date(date(2026, 9, 8), 'path/to/season_pa.jsonl')
picks = slate['picks_6']  # List of 6 highest-prob hitters
```

### `render_html`
Render to HTML:
```python
html = render_html(slate_data)
```

## Calibration & Validation

Metrics are produced by a genuine walk-forward split: the model is fitted on
plate appearances before a cutoff date (80% of the season) and scored on the
~35,000 that came after. No outcome in the test window is visible during
fitting. Results are printed at startup and shown in the web UI.

**Read AUC first.** The slate ranks hitters and takes the top six, so ordering
is what matters. Per-PA Brier and log loss are dominated by the ~3% base rate
and barely separate the models.

| Prior | AUC | Top-decile lift | Brier | Log loss | Cal. RMSE |
|---|---|---|---|---|---|
| Flat league rate | 0.5641 | 1.47x | 0.02832 | 0.13132 | 0.00200 |
| **KNN comparables** | **0.5780** | **1.49x** | 0.02835 | 0.13201 | 0.00195 |
| SVR | 0.5665 | 1.41x | 0.02857 | 0.13504 | 0.08531 |
| Ensemble (served) | 0.5773 | 1.48x | 0.02835 | 0.13209 | 0.00199 |

The comparables prior is what earns the ranking improvement (AUC 0.564 to
0.578). Calibration is a wash, which is expected: for a regular with 600 PA the
observed rate swamps any prior. Numbers above are from the 2026 season through
Sep 8 and will move as data accumulates.

## Troubleshooting

### No games scheduled
If the web interface shows "No games scheduled," check that today's date has games in the MLB schedule. The model uses `schedule()` to find games with state "Scheduled" or "Final".

### Data fetch errors
Network timeouts are retried with exponential backoff (up to 4 attempts). Check your internet connection or run with existing cached data.

### Docker issues
```bash
# Clean rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up
```

## Future Enhancements

- [ ] Walk-forward backtesting with Brier score tracking
- [ ] Injury updates via ESPN API integration
- [ ] Pitcher bullpen depth scoring
- [ ] Weather adjustments (wind, humidity, temp)
- [x] ML models (SVM, KNN, random forest, XGBoost, neural net) for non-linear effects
- [x] REST API for programmatic access (`/api/remaining`)

## License

Built for research and educational purposes. MLB data via official MLB Stats API.

## Questions?

Open an issue or check the source code comments for implementation details.
