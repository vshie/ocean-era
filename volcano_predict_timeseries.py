#!/usr/bin/env python3
"""Re-measure slopes from USGS PNGs and generate an extrapolation dataset/plot.

Inputs:
- 30-day plot: used for BLUE raw series (digitized) and BLUE slope (post-drop trough -> now)
- 90-day plot: used for RED threshold slope (fit through left+right pre-drop peaks)

Assumptions:
- 30-day plot spans 30 days (x)
- 90-day plot spans 90 days (x)
- 30-day y-grid spacing: 5 microrad per horizontal gridline
- 90-day y-grid spacing: 10 microrad per horizontal gridline
- We anchor absolute y-units by setting BLUE at the right edge to blue_now_u (default -13.5 µrad)

Outputs:
- CSV of raw blue (last 30d) + extrapolated lines out to crossing+1day
- PNG forecast plot
- Overlay PNGs for the fit lines on their source plots

Usage:
  python volcano_predict_timeseries.py \
    --month volcano/UWD-POC-TILT-month.png \
    --three-month volcano/UWD-TILT-3month.png \
    --blue-now -13.5 \
    --out-prefix volcano/predict
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from volcano_threshold_overlay_v2 import (
    CROP,
    extract_blue_trace,
    last_real_blue_index,
    moving_average,
    detect_drops,
    peaks_before_drops,
    line_through,
)


@dataclass
class Line:
    m: float
    b: float

    def y(self, t: float) -> float:
        return self.m * t + self.b


def detect_hgrid_spacing_px(im: Image.Image) -> float:
    x0, y0, x1, y1 = CROP
    sub = np.array(im.convert("RGB"))[y0:y1, x0:x1]
    r, g, b = sub[:, :, 0].astype(int), sub[:, :, 1].astype(int), sub[:, :, 2].astype(int)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    ch = mx - mn
    br = (r + g + b) / 3
    mask = (ch < 20) & (br > 120) & (br < 230)
    row_counts = mask.sum(axis=1)
    thr = row_counts.mean() + 3 * row_counts.std()
    rows = np.where(row_counts > thr)[0]

    centers = []
    if rows.size:
        s = rows[0]
        p = rows[0]
        for rr in rows[1:]:
            if rr == p + 1:
                p = rr
            else:
                centers.append((s + p) / 2)
                s = rr
                p = rr
        centers.append((s + p) / 2)

    if len(centers) < 2:
        return float("nan")
    return float(np.median(np.diff(centers)))


def blue_pixels_to_microrad(y_px: np.ndarray, *, y_now_px: float, blue_now_u: float, microrad_per_grid: float, grid_spacing_px: float) -> np.ndarray:
    # Positive microradians are upward on the plot, which is negative pixel-y.
    u_per_px = microrad_per_grid / grid_spacing_px
    return blue_now_u - (y_px - y_now_px) * u_per_px


def overlay_three_month_threshold(im: Image.Image, thr_fit, out_path: str):
    x0, y0, x1, y1 = CROP
    w = x1 - x0

    out = im.convert("RGBA")
    d = ImageDraw.Draw(out, "RGBA")

    xs = np.linspace(x0, x1, 250)
    t_days = (xs - x0) / (w - 1) * 90.0
    ys = thr_fit.m * t_days + thr_fit.b
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
    d.line(pts, fill=(255, 0, 0, 210), width=3)

    d.text((12, 10), "3-month: fitted threshold (red)", fill=(255, 255, 255, 235))
    out.save(out_path)


def overlay_month_cyan(im: Image.Image, x_tr, y_tr, x_now, y_now, out_path: str):
    """Overlay the cyan ramp fit, clipped to the rightmost blue data point."""

    x0, y0, x1, y1 = CROP
    x_right = min(float(x_now), float(x1 - 1))
    y_right = y_now

    out = im.convert("RGBA")
    d = ImageDraw.Draw(out, "RGBA")

    d.line([(x_tr, y_tr), (x_right, y_right)], fill=(0, 255, 255, 220), width=3)

    r = 6
    for xx, yy in [(x_tr, y_tr), (x_right, y_right)]:
        d.ellipse((xx - r, yy - r, xx + r, yy + r), outline=(0, 255, 255, 230), width=3)

    d.text((12, 10), "30-day: trough→now ramp fit (cyan)", fill=(255, 255, 255, 235))
    out.save(out_path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--three-month", required=True)
    ap.add_argument("--blue-now", type=float, default=-13.5)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args(argv)

    im_m = Image.open(args.month)
    im_3 = Image.open(args.three_month)

    # Digitize blue traces
    xm, ym = extract_blue_trace(im_m)
    x3, y3 = extract_blue_trace(im_3)
    ym = moving_average(ym, 11)
    y3 = moving_average(y3, 11)

    x0, y0, x1, y1 = CROP
    w = x1 - x0

    # X -> days
    tm_d = (xm - x0) / (w - 1) * 30.0
    t3_d = (x3 - x0) / (w - 1) * 90.0

    # Grid spacing to map pixels->microrad
    m_grid_px = detect_hgrid_spacing_px(im_m)
    th_grid_px = detect_hgrid_spacing_px(im_3)

    # Anchor absolute units at right edge: blue_now
    y_m_now_px = float(np.nanmean(ym[-20:]))
    y_3_now_px = float(np.nanmean(y3[-20:]))

    blue_m_u = blue_pixels_to_microrad(ym, y_now_px=y_m_now_px, blue_now_u=args.blue_now, microrad_per_grid=5.0, grid_spacing_px=m_grid_px)
    blue_3_u = blue_pixels_to_microrad(y3, y_now_px=y_3_now_px, blue_now_u=args.blue_now, microrad_per_grid=10.0, grid_spacing_px=th_grid_px)

    # --- 3-month threshold fit in pixel space (left+right peak-before-drop anchors) ---
    drops3 = detect_drops(y3)
    peaks3 = peaks_before_drops(x3, t3_d, y3, drops3, wback=180)
    peaks3_sorted = sorted(peaks3, key=lambda p: p[1])
    pL = peaks3_sorted[0]
    pR = peaks3_sorted[-1]
    thr_px = line_through(pL[1], pL[2], pR[1], pR[2])  # y_px = m*t_days + b

    # Convert threshold line to microrad units using same 3-month mapping
    # u = blue_now - (y - y_now)*u_per_px
    u_per_px_3 = 10.0 / th_grid_px

    def thr_u(t_days: float) -> float:
        ypx = thr_px.y(t_days)
        return args.blue_now - (ypx - y_3_now_px) * u_per_px_3

    # Threshold slope in µrad/day from fit
    # u(t)=blue_now - (m*t+b - y_now)*u_per_px => du/dt = -(m)*u_per_px
    thr_u_slope = -thr_px.m * u_per_px_3
    thr_u_now = thr_u(90.0)

    # --- 30-day cyan slope: trough after most recent drop to now ---
    dy = np.diff(ym)
    thr = np.nanpercentile(dy, 99) * 0.6
    cand = np.where(dy > thr)[0]
    drops = []
    last = -999
    for i in cand.tolist():
        if i - last > 10:
            drops.append(i)
            last = i
    last_drop = drops[-1] if drops else int(0.8 * len(ym))

    px_per_day = w / 30.0
    win_px = int(max(30, 4 * px_per_day))
    lo = min(len(ym) - 2, last_drop + 3)
    hi = min(len(ym), lo + win_px)
    idx_tr = lo + int(np.nanargmax(ym[lo:hi]))

    # endpoints in µrad
    u_tr = float(blue_m_u[idx_tr])
    d_tr = float(tm_d[idx_tr])
    u_now = float(args.blue_now)
    d_now = float(tm_d[-1])
    cyan_u_slope = (u_now - u_tr) / (d_now - d_tr)

    # Intersection time (days from now) where cyan and red meet:
    # u_now + cyan*(t) = thr_now + thr_slope*(t)
    denom = (cyan_u_slope - thr_u_slope)
    t_cross_days = (thr_u_now - u_now) / denom if abs(denom) > 1e-9 else float("nan")

    # Build future dataset from now to cross+1 day
    horizon_days = max(1.0, t_cross_days + 1.0)
    t_future = np.linspace(0.0, horizon_days, int(horizon_days * 24) + 1)  # hourly
    cyan_future = u_now + cyan_u_slope * t_future
    thr_future = thr_u_now + thr_u_slope * t_future

    # Write CSV
    out_csv = args.out_prefix + ".csv"
    with open(out_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["t_days_from_now", "blue_raw_u", "cyan_u", "threshold_u"])
        # raw blue for last 30 days in relative time: t_from_now = tm_d - 30
        for t, u in zip(tm_d, blue_m_u):
            wcsv.writerow([t - 30.0, u, "", ""])
        for t, u1, u2 in zip(t_future, cyan_future, thr_future):
            wcsv.writerow([t, "", u1, u2])

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(1, 1, 1)

    # plot raw blue last 30d, in absolute relative time -30..0
    ax.plot(tm_d - 30.0, blue_m_u, color="#1f77b4", linewidth=1.5, label="Blue (digitized, last 30d)")

    ax.plot(t_future, cyan_future, color="#00cfe3", linewidth=2.5, label=f"Cyan trend (+{cyan_u_slope:.2f} µrad/day)")
    ax.plot(t_future, thr_future, color="#d62728", linewidth=2.5, label=f"Red threshold ({thr_u_slope:+.2f} µrad/day)")

    if np.isfinite(t_cross_days):
        ax.axvline(t_cross_days, color="k", alpha=0.25, linestyle="--")
        ax.scatter([t_cross_days], [u_now + cyan_u_slope * t_cross_days], color="k", s=25)
        ax.text(t_cross_days + 0.2, u_now + cyan_u_slope * t_cross_days, f"cross ≈ {t_cross_days:.2f} d", fontsize=9)

    ax.axvline(0, color="k", alpha=0.15)
    ax.set_xlabel("Days from now (0 = right edge of plot)")
    ax.set_ylabel("Tilt (µrad, anchored at blue_now)")
    ax.set_title("Kīlauea tilt: blue raw + trendlines to intersection")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    out_png = args.out_prefix + ".png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # Overlays
    out_thr = args.out_prefix + "_3month_threshold_overlay.png"
    out_cyan = args.out_prefix + "_30day_cyan_overlay.png"
    overlay_three_month_threshold(im_3, Line(thr_px.m, thr_px.c), out_thr)
    # Use the rightmost actual blue pixel (not interpolated) as "now".
    idx_now = last_real_blue_index(im_m)
    overlay_month_cyan(
        im_m,
        float(xm[idx_tr]),
        float(ym[idx_tr]),
        float(xm[idx_now]),
        float(ym[idx_now]),
        out_cyan,
    )

    print("blue_now_u", u_now)
    print("threshold_now_u", thr_u_now)
    print("cyan_slope_u_per_day", cyan_u_slope)
    print("red_slope_u_per_day", thr_u_slope)
    print("t_cross_days", t_cross_days, "hours", t_cross_days * 24)
    print("Wrote:", out_csv)
    print("Wrote:", out_png)
    print("Wrote:", out_thr)
    print("Wrote:", out_cyan)


if __name__ == "__main__":
    raise SystemExit(main())
