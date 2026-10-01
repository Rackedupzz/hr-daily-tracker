"""Retired: the HR probability calibration is refitted by the season replay.

This used to refit one logistic recalibration (ensemble.CAL_INTERCEPT /
CAL_SLOPE) on rolling-backtest rows. Those rows lacked the live projection's
park, weather and pitcher terms and counted pinch hitters as starters, so the
constants were fitted to a different distribution than the one served -- in
September they squeezed the top of the slate to 16% while it homered at
19-21%. Since 2026-09-30 the served probability comes from the stacked model
and its top calibration (ensemble.HR_STACK / HR_TOP_CAL), fitted and validated
walk-forward on a day-by-day replay of the live projection. This entry point
runs that replay, so the old command keeps working:

    python -m mlb_hr.calibration_fit   ==   python -m mlb_hr.replay
"""
from mlb_hr.replay import main

if __name__ == "__main__":
    main()
