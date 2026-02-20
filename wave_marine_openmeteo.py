#!/usr/bin/env python3
"""Fetch waves/swell for a lat/lon using Open-Meteo Marine Weather API and generate a plot.
 
Outputs:
  - CSV: time series of selected marine variables
  - PNG: multi-panel plot (wave + swell + wind-wave heights, periods, directions)
 
Example:
  python wave_marine_openmeteo.py --lat 19.727588 --lon -156.0627484 --tz Pacific/Honolulu \
    --past-days 5 --future-days 5 --title "Keahole Point" --out-prefix waves_keahole
"""
 
from __future__ import annotations
 
import argparse
import csv
import datetime as dt
import json
import math
import urllib.parse
import urllib.request
 
 
def http_get_json(url: str, timeout_s: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "clawdbot-marine/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))
 
 
def build_openmeteo_marine_url(
    *,
    lat: float,
    lon: float,
    timezone: str,
    past_days: int,
    forecast_days: int,
) -> str:
    base = "https://marine-api.open-meteo.com/v1/marine"
 
    # Keep the set small/clear for a first plot.
    hourly = [
        "wave_height",
        "wave_direction",
        "wave_period",
        "swell_wave_height",
        "swell_wave_direction",
        "swell_wave_period",
        "wind_wave_height",
        "wind_wave_direction",
        "wind_wave_period",
        "ocean_current_velocity",
        "ocean_current_direction",
    ]
 
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "hourly": ",".join(hourly),
        "timezone": timezone,
        "length_unit": "imperial",  # feet
        "velocity_unit": "mph",     # for currents
        "past_days": str(int(past_days)),
        "forecast_days": str(int(forecast_days)),
    }
 
    return base + "?" + urllib.parse.urlencode(params)
 
 
def write_csv(times: list[str], series: dict[str, list], out_csv: str):
    # Standardize field order
    fields = ["time"] + [k for k in series.keys()]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, t in enumerate(times):
            row = {"time": t}
            for k, vals in series.items():
                row[k] = vals[i] if i < len(vals) else None
            w.writerow(row)
 
 
def _polar_to_uv(mag, deg_toward):
    """Convert (magnitude, direction degrees toward which it goes) to u/v.
 
    Works with scalars or numpy arrays.
    Assumes degrees: 0=N, 90=E.
    u:+east, v:+north
    """
    try:
        import numpy as np
 
        r = np.deg2rad(deg_toward)
        u = mag * np.sin(r)
        v = mag * np.cos(r)
        return u, v
    except Exception:
        r = math.radians(float(deg_toward))
        u = float(mag) * math.sin(r)
        v = float(mag) * math.cos(r)
        return u, v
 
 
def plot(times: list[str], series: dict[str, list], out_png: str, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.dates as mdates
    except Exception as e:
        raise RuntimeError(
            "Plotting requires matplotlib (and numpy). Install them. "
            f"Original error: {e}"
        )
 
    t_dt = [dt.datetime.fromisoformat(t) for t in times]
    t_num = mdates.date2num(t_dt)
 
    def arr(name: str):
        return np.array(series.get(name, [np.nan] * len(times)), dtype=float)
 
    wave_h = arr("wave_height")
    swell_h = arr("swell_wave_height")
    windwave_h = arr("wind_wave_height")
 
    wave_p = arr("wave_period")
    swell_p = arr("swell_wave_period")
    windwave_p = arr("wind_wave_period")
 
    # Directions for quiver overlays (degrees)
    wave_dir = arr("wave_direction")
    swell_dir = arr("swell_wave_direction")
 
    # Current vectors
    cur_v = arr("ocean_current_velocity")
    cur_dir = arr("ocean_current_direction")
    cur_u, cur_vv = _polar_to_uv(cur_v, cur_dir)
 
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.5, 1.2], hspace=0.25)
 
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
 
    # Panel 1: Heights (ft)
    ax1.plot(t_dt, wave_h, label="Wave height (ft)", linewidth=2)
    ax1.plot(t_dt, swell_h, label="Swell height (ft)", linewidth=1.7)
    ax1.plot(t_dt, windwave_h, label="Wind-wave height (ft)", linewidth=1.7)
    ax1.set_ylabel("Height (ft)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")
 
    # Quiver overlay for dominant SWELL direction: show direction only (unit arrows)
    # Subsample to keep readable
    step = max(1, int(len(t_dt) / 36))
    tq = np.array(t_num)[::step]
    # Put the arrows near the top of panel
    yq = np.nanmax(wave_h) * 0.95 if np.isfinite(np.nanmax(wave_h)) else 1.0
    yq_arr = np.full_like(tq, yq, dtype=float)
    udir, vdir = _polar_to_uv(np.ones_like(tq), swell_dir[::step])
    ax1.quiver(
        mdates.num2date(tq),
        yq_arr,
        udir,
        vdir,
        color="gray",
        alpha=0.55,
        scale=25,
        width=0.002,
        headwidth=3,
        headlength=4,
    )
    ax1.text(
        0.01,
        0.02,
        "Gray arrows (top): dominant swell direction (toward)",
        transform=ax1.transAxes,
        fontsize=9,
        alpha=0.7,
    )
 
    # Panel 2: Periods (s)
    ax2.plot(t_dt, wave_p, label="Wave period (s)", linewidth=2)
    ax2.plot(t_dt, swell_p, label="Swell period (s)", linewidth=1.7)
    ax2.plot(t_dt, windwave_p, label="Wind-wave period (s)", linewidth=1.7)
    ax2.set_ylabel("Period (s)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right")
 
    # Panel 3: Surface currents (mph) as u/v quiver along baseline
    ax3.plot(t_dt, cur_v, label="Surface current speed (mph)", linewidth=2, color="#7b2cbf")
    ax3.set_ylabel("Current (mph)")
    ax3.grid(True, alpha=0.25)
 
    # Direction quiver for currents (unit arrows near top of panel 3)
    yq3 = np.nanmax(cur_v) * 0.9 if np.isfinite(np.nanmax(cur_v)) else 1.0
    yq3_arr = np.full_like(tq, yq3, dtype=float)
    ucur, vcur = _polar_to_uv(np.ones_like(tq), cur_dir[::step])
    ax3.quiver(
        mdates.num2date(tq),
        yq3_arr,
        ucur,
        vcur,
        color="#7b2cbf",
        alpha=0.55,
        scale=25,
        width=0.002,
        headwidth=3,
        headlength=4,
    )
    ax3.text(
        0.01,
        0.02,
        "Arrows (bottom): surface current direction (toward)",
        transform=ax3.transAxes,
        fontsize=9,
        alpha=0.7,
        color="#7b2cbf",
    )
 
    ax3.legend(loc="upper right")
 
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax3.xaxis.get_major_locator()))
 
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
 
 
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--tz", type=str, default="Pacific/Honolulu")
    ap.add_argument("--past-days", type=int, default=5)
    ap.add_argument("--future-days", type=int, default=5)
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--out-prefix", type=str, default="marine")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
 
    # forecast_days includes today; so for "future N days" we request N+1 including today.
    forecast_days = int(args.future_days) + 1
 
    url = build_openmeteo_marine_url(
        lat=args.lat,
        lon=args.lon,
        timezone=args.tz,
        past_days=int(args.past_days),
        forecast_days=forecast_days,
    )
 
    payload = http_get_json(url)
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise RuntimeError(f"No hourly time series returned. Keys={list(hourly.keys())}")
 
    series = {k: v for k, v in hourly.items() if k != "time"}
 
    out_csv = f"{args.out_prefix}.csv"
    out_png = f"{args.out_prefix}.png"
 
    write_csv(times, series, out_csv)
 
    if not args.no_plot:
        title = args.title or f"Marine (Open-Meteo) @ {args.lat:.4f}, {args.lon:.4f}"
        plot(times, series, out_png, title)
 
    print(f"Wrote: {out_csv}")
    if not args.no_plot:
        print(f"Wrote: {out_png}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))