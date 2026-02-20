#!/usr/bin/env python3
"""Ocean Era (or any ocean point): next-N-days plot showing wind, swell, surface current,
plus an alignment indicator (parallel/orthogonal/opposite).

Data sources (via Open-Meteo):
- Wind: https://api.open-meteo.com/v1/forecast (10 m wind)
- Swell + Currents: https://marine-api.open-meteo.com/v1/marine

Notes on direction conventions:
- Open-Meteo winddirection_* is meteorological degrees *FROM* which the wind blows.
  We convert to direction *TOWARD* for alignment.
- Marine wave/swell/current directions are treated as degrees *TOWARD* (0=N, 90=E).

Example:
  python ocean_era_alignment.py --lat 19.829 --lon -156.1208 --tz Pacific/Honolulu \
    --days 10 --title "Ocean Era Site – 10-day Conditions + Alignment" --out-prefix oceanera_10d
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import urllib.parse
import urllib.request


def http_get_json(url: str, timeout_s: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "clawdbot-oceanera/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def build_wind_url(*, lat: float, lon: float, tz: str, days: int) -> str:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "hourly": "windspeed_10m,winddirection_10m",
        "wind_speed_unit": "mph",
        "timezone": tz,
        "forecast_days": str(int(days) + 1),  # include today
    }
    return base + "?" + urllib.parse.urlencode(params)


def build_marine_url(*, lat: float, lon: float, tz: str, days: int) -> str:
    base = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "hourly": ",".join(
            [
                "swell_wave_height",
                "swell_wave_direction",
                "swell_wave_period",
                "ocean_current_velocity",
                "ocean_current_direction",
            ]
        ),
        "timezone": tz,
        "length_unit": "imperial",  # feet
        "velocity_unit": "mph",
        "forecast_days": str(int(days) + 1),
    }
    return base + "?" + urllib.parse.urlencode(params)


def deg_to_unit_uv_toward(deg_toward):
    """0°=North, 90°=East. Returns (u_east, v_north). Works for scalars/arrays."""
    try:
        import numpy as np

        r = np.deg2rad(deg_toward)
        return np.sin(r), np.cos(r)
    except Exception:
        r = math.radians(float(deg_toward))
        return math.sin(r), math.cos(r)


def wind_from_to_toward_deg(deg_from):
    try:
        import numpy as np

        return (deg_from + 180.0) % 360.0
    except Exception:
        return (float(deg_from) + 180.0) % 360.0


def classify_alignment(c):
    # thresholds based on cos(angle)
    # parallel: <=45°  (cos >= 0.707)
    # orthogonal: ~90° (|cos| <= 0.259)
    # opposite: >=135° (cos <= -0.707)
    if c >= 0.707:
        return "parallel"
    if c <= -0.707:
        return "opposite"
    if abs(c) <= 0.259:
        return "orthogonal"
    return "mixed"


def plot(times, wind_speed, wind_dir_from, swell_h, swell_dir, cur_speed, cur_dir, out_png, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.dates as mdates
    except Exception as e:
        raise RuntimeError(f"Plotting requires matplotlib/numpy. {e}")

    t_dt = [dt.datetime.fromisoformat(t) for t in times]

    wind_tow = wind_from_to_toward_deg(np.array(wind_dir_from, dtype=float))

    # Unit vectors (toward)
    uw, vw = deg_to_unit_uv_toward(wind_tow)
    us, vs = deg_to_unit_uv_toward(np.array(swell_dir, dtype=float))
    uc, vc = deg_to_unit_uv_toward(np.array(cur_dir, dtype=float))

    # Pairwise cos similarities
    cos_ws = uw * us + vw * vs
    cos_wc = uw * uc + vw * vc
    cos_sc = us * uc + vs * vc
    align = (cos_ws + cos_wc + cos_sc) / 3.0

    # Category for coloring
    cats = np.array([classify_alignment(float(x)) for x in align])
    cat_color = {
        "parallel": "#2a9d8f",
        "orthogonal": "#e9c46a",
        "opposite": "#e76f51",
        "mixed": "#8d99ae",
    }
    colors = [cat_color[c] for c in cats]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.2, 1.2, 0.9], hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)

    # Magnitude panels + direction quiver overlays (unit arrows near the top)
    wind_color = "#264653"
    swell_color = "#1d3557"
    cur_color = "#7b2cbf"

    ax1.plot(t_dt, wind_speed, color=wind_color, linewidth=2, label="Wind speed 10 m (mph)")
    ax1.set_ylabel("Wind (mph)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")

    ax2.plot(t_dt, swell_h, color=swell_color, linewidth=2, label="Swell height (ft)")
    ax2.set_ylabel("Swell (ft)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right")

    ax3.plot(t_dt, cur_speed, color=cur_color, linewidth=2, label="Surface current speed (mph)")
    ax3.set_ylabel("Current (mph)")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper right")

    # Subsample arrows to keep readable
    step = max(1, int(len(t_dt) / 36))
    tq = np.array(mdates.date2num(t_dt))[::step]
    tqd = mdates.num2date(tq)

    def quiver_on(ax, y_series, u_unit, v_unit, color, label_text):
        y_series = np.array(y_series, dtype=float)
        ytop = np.nanmax(y_series)
        yq = ytop * 0.9 if np.isfinite(ytop) and ytop != 0 else 1.0
        yq_arr = np.full(len(tqd), yq, dtype=float)
        ax.quiver(
            tqd,
            yq_arr,
            np.array(u_unit)[::step],
            np.array(v_unit)[::step],
            color=color,
            alpha=0.55,
            scale=25,
            width=0.002,
            headwidth=3,
            headlength=4,
        )
        ax.text(0.01, 0.03, label_text, transform=ax.transAxes, fontsize=9, alpha=0.75, color=color)

    # Wind: convert FROM to TOWARD, then arrow direction is TOWARD
    quiver_on(ax1, wind_speed, uw, vw, wind_color, "Arrows: wind direction (toward)")
    quiver_on(ax2, swell_h, us, vs, swell_color, "Arrows: swell direction (toward)")
    quiver_on(ax3, cur_speed, uc, vc, cur_color, "Arrows: surface current direction (toward)")

    # Alignment panel: value + categorical coloring
    ax4.scatter(t_dt, align, c=colors, s=18, label="Hourly alignment")

    # Rolling mean (helps highlight the best window)
    win_hours = 6
    if len(align) >= win_hours:
        kernel = np.ones(win_hours) / float(win_hours)
        align_smooth = np.convolve(align, kernel, mode="same")
        ax4.plot(t_dt, align_smooth, color="#000000", alpha=0.35, linewidth=2, label=f"{win_hours}h mean")
    else:
        align_smooth = align

    # Pick the best future time (max of smoothed alignment, ignoring the past)
    now = dt.datetime.now(dt.timezone.utc)
    # naive timestamps from API are in local tz; interpret as local and compare by converting to naive local now
    try:
        local_now = dt.datetime.now()
        future_mask = np.array([t >= local_now for t in t_dt])
    except Exception:
        future_mask = np.ones(len(t_dt), dtype=bool)

    if np.any(future_mask):
        idx = np.argmax(np.where(future_mask, align_smooth, -np.inf))
        best_t = t_dt[int(idx)]
        best_v = float(align_smooth[int(idx)])
        ax4.axvline(best_t, color="#000000", alpha=0.25, linewidth=2)
        ax4.annotate(
            f"Best window: {best_t:%a %m/%d %H:%M}  (score {best_v:+.2f})",
            xy=(best_t, best_v),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
        )

    ax4.axhline(0.707, color=cat_color["parallel"], alpha=0.25, linewidth=1)
    ax4.axhline(-0.707, color=cat_color["opposite"], alpha=0.25, linewidth=1)
    ax4.axhline(0.0, color="black", alpha=0.15, linewidth=1)
    ax4.set_ylim(-1.05, 1.05)
    ax4.set_ylabel("Alignment\n(mean cos)")
    ax4.grid(True, alpha=0.25)

    ax4.legend(loc="upper right", fontsize=9, framealpha=0.6)
    ax4.text(0.01, 0.03, "Green≈parallel • Yellow≈orthogonal • Red≈opposite • Gray≈mixed", transform=ax4.transAxes, fontsize=9, alpha=0.8)

    ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax4.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax4.xaxis.get_major_locator()))

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--tz", type=str, default="Pacific/Honolulu")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--out-prefix", type=str, default="alignment")
    args = ap.parse_args(argv)

    wind = http_get_json(build_wind_url(lat=args.lat, lon=args.lon, tz=args.tz, days=args.days))
    marine = http_get_json(build_marine_url(lat=args.lat, lon=args.lon, tz=args.tz, days=args.days))

    wt = wind["hourly"]["time"]
    mt = marine["hourly"]["time"]

    # Align by time key intersection (should match, but be safe)
    widx = {t: i for i, t in enumerate(wt)}
    midx = {t: i for i, t in enumerate(mt)}
    times = sorted(set(widx).intersection(midx))

    def w(name):
        arr = wind["hourly"].get(name, [])
        return [arr[widx[t]] if widx[t] < len(arr) else None for t in times]

    def m(name):
        arr = marine["hourly"].get(name, [])
        return [arr[midx[t]] if midx[t] < len(arr) else None for t in times]

    wind_speed = w("windspeed_10m")
    wind_dir = w("winddirection_10m")

    swell_h = m("swell_wave_height")
    swell_dir = m("swell_wave_direction")

    cur_speed = m("ocean_current_velocity")
    cur_dir = m("ocean_current_direction")

    out_png = f"{args.out_prefix}.png"
    title = args.title or f"Conditions + Alignment @ {args.lat:.4f}, {args.lon:.4f}"
    plot(times, wind_speed, wind_dir, swell_h, swell_dir, cur_speed, cur_dir, out_png, title)

    print(f"Wrote: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))