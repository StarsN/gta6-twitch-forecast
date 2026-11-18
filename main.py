"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Entry for make.fun's gta6-twitch-ccu-forecast-01. The program reads one case
file (schema "makefun/twitch-ccu-case/v1") and writes, for each of hours 2
to 24 of the release day, five quantiles (5%, 25%, 50%, 75%, 95%) of that
hour's concurrent Twitch viewers in the game's category.

How it forecasts
----------------
The case's `known` row gives hour 1's print, p1. Everything else is a
shape: for each later hour h, the forecast is p1 times a ratio, and the
ratio is learned from the case's own `comparables` -- the release-day
curves of past big game launches (SullyGnome hourly Twitch category
averages, 14 days before to 3 days after each launch).

Two facts about launch days drive the model, both measured on those
comparables:

1. Console-only launches that unlock at midnight local time share one
   shape: viewers dip to ~0.6x hour 1 in the small hours (hours 3-5),
   then climb as the US unlocks and prime time arrives, peaking at
   ~2.4x hour 1 around hours 19-20. PC-simultaneous launches look
   completely different (their hour 1 already holds the global PC
   unlock), so the model only learns from console-only launches --
   ones whose `platforms` name no PC storefront. In the final case
   that pool is the 8 console launches in the kit plus the 4 practice
   launches (all console-only), 12 in all.

2. The shape depends on the launch's scale: across console launches,
   log(hour-1 print) correlates about -0.85 with the peak log-ratio.
   A launch already huge at hour 1 has less headroom left; a small
   hour 1 leaves room for a multi-fold climb. So for each hour the
   model fits a one-predictor regression of log(viewers_h / viewers_1)
   on log(viewers_1) across the console pool, and reads the median
   off that line at the case's own p1. The slopes are ~0 in the early
   hours and about -0.3 in the late hours, exactly where saturation
   bites. The spread at each hour is the regression's residual
   standard deviation (floored at 0.15 in log units), so the 5% and
   95% quantiles are honest about how much launch days differ.

The program runs offline on the standard library only, is fully
deterministic (the pool is sorted before every sum, no randomness,
no clock, no network), and runs in well under a second per case.

Usage (the grader runs it with /input and /output; locally set the
environment variables instead):

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path; put it back
# first so a module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The house baseline's spread (log units per sqrt hour), used only if a
# case carries too few comparables to fit the regression at all.
FALLBACK_SIGMA = 0.15
# A regression spread below this is not trusted: with a handful of
# launches the residual standard deviation is a noisy estimate.
SPREAD_FLOOR = 0.15
# A launch joins the console pool only when its platforms string names
# no PC storefront.
PC_MARKERS = ("pc", "steam", "stadia", "epic")


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as the starter rounds it.

    A whole number comes back as an int, so json.dumps writes 1240
    where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day(comparable: dict) -> list | None:
    """The comparable's 24 hourly viewer counts for its release day.

    Returns None when the day is incomplete (the target's own history
    comparable, for example, has no release day yet) or hour 1 is
    missing or zero.
    """
    try:
        day_start = parse_ts(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour: dict[int, float] = {}
    for point in comparable.get("hourly") or []:
        try:
            when = parse_ts(point["from"])
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        hour = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24 and isinstance(viewers, (int, float)):
            by_hour[hour] = float(viewers)
    if len(by_hour) != 24:
        return None
    day = [by_hour[h] for h in range(1, 25)]
    if day[0] < 1:
        return None
    return day


def is_console_only(comparable: dict) -> bool:
    """True when the launch's platforms name no PC storefront."""
    platforms = comparable.get("platforms")
    if not isinstance(platforms, str):
        return False
    lowered = platforms.lower()
    return not any(marker in lowered for marker in PC_MARKERS)


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least squares line through (xs, ys): returns (intercept, slope)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom if denom else 0.0
    return mean_y - slope * mean_x, slope


def fallback_quantiles(p1: float, hour: int) -> list[float | int]:
    """The house persistence baseline: every hour stays at hour 1."""
    return [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1))) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    known = case.get("known") or []
    first = next((row for row in known if row.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    # The pool: every comparable with a complete release day, sorted so
    # every sum below is byte-for-byte reproducible.
    launches = []
    for comparable in case.get("comparables") or []:
        day = release_day(comparable)
        if day is None:
            continue
        launches.append(
            {
                "name": str(comparable.get("name", "")),
                "day_start": str(comparable.get("day_start", "")),
                "console": is_console_only(comparable),
                "day": day,
                "h1": day[0],
            }
        )
    launches.sort(key=lambda launch: (launch["name"], launch["day_start"]))

    console = [launch for launch in launches if launch["console"]]
    big = [launch for launch in launches if launch["h1"] >= 20000]
    if len(console) >= 4:
        pool = console
    elif len(big) >= 4:
        pool = big
    else:
        pool = launches
    if len(pool) < 3:
        return {
            "hours": [
                {"hour": row["hour"], "q": fallback_quantiles(p1, row["hour"])}
                for row in case["forecast"]
            ]
        }

    log_p1 = math.log(p1)
    log_h1 = [math.log(launch["h1"]) for launch in pool]
    hours = []
    for row in case["forecast"]:
        hour = row["hour"]
        log_ratios = [math.log(launch["day"][hour - 1] / launch["h1"]) for launch in pool]
        intercept, slope = ols(log_h1, log_ratios)
        median_log_ratio = intercept + slope * log_p1
        residuals = [
            ratio - (intercept + slope * x) for ratio, x in zip(log_ratios, log_h1)
        ]
        spread = math.sqrt(sum(r * r for r in residuals) / max(1, len(residuals) - 2))
        spread = max(spread, SPREAD_FLOOR)
        quantiles = []
        for z in Z:
            value = p1 * math.exp(median_log_ratio + z * spread)
            quantiles.append(round2(min(max(value, 1.0), 1_000_000_000.0)))
        # Rounding can only move a quantile by a cent; keep the row
        # non-decreasing no matter what.
        for index in range(1, len(quantiles)):
            if quantiles[index] < quantiles[index - 1]:
                quantiles[index] = quantiles[index - 1]
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
