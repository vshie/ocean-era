#!/usr/bin/env python3
"""Affine y-axis mapping between USGS Kilauea tilt plots + overlay threshold on 2-day plot.

Goal:
- Fit drifting threshold trend line on 3-month plot (red).
- Fit overall blue trend on 2-day plot (cyan).
- Map the 3-month y-scale onto the 2-day y-scale using two shared features:
  1) right-edge (most recent) blue value
  2) most-recent post-drop trough value

This yields an affine map: y_2day = a*y_3month + b.
Then we can map the red threshold line onto the 2-day plot and estimate the next
intersection time (nearest hour) in a best-effort way.

PNG-only digitizing; results are approximate.

Usage:
  python volcano_affine_predict_overlay.py \
    --two-day volcano/UWD-TILT-2day.png \
    --three-month volcano/UWD-TILT-3month.png \
    --out-two volcano/UWD-TILT-2day_mapped_threshold.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from volcano_threshold_overlay_v2 import (
    CROP,
    extract_blue_trace,
    moving_average,
    detect_drops,
    peaks_before_drops,
    line_through,
)


@dataclass
class Line:
    m: float
    c: float

    def y(self, t: float) -> float:
        return self.m * t + self.c


def y_trough_after_drop(y, drop_idx: int, *, lo_off=5, hi_off=80):
    """Estimate trough (post-drop minimum in *data* space = max pixel-y) after a drop."""
    n = len(y)
    lo = min(n - 1, max(0, drop_idx + lo_off))
    hi = min(n, max(lo + 5, drop_idx + hi_off))
    seg = np.asarray(y[lo:hi], dtype=float)
    if seg.size == 0:
        return float(np.nan)
    # robust high percentile instead of max
    return float(np.nanpercentile(seg, 90))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-day", required=True)
    ap.add_argument("--three-month", required=True)
    ap.add_argument("--out-two", required=True)
    args = ap.parse_args(argv)

    im2 = Image.open(args.two_day)
    im3 = Image.open(args.three_month)

    x2, y2 = extract_blue_trace(im2)
    x3, y3 = extract_blue_trace(im3)

    y2 = moving_average(y2, 9)
    y3 = moving_average(y3, 11)

    x0, y0, x1, y1 = CROP
    w = x1 - x0

    t2_h = (x2 - x0) / (w - 1) * 48.0
    t3_d = (x3 - x0) / (w - 1) * 90.0

    # --- 3-month threshold line (red): anchored left+right peak-before-drop points ---
    drops3 = detect_drops(y3)
    peaks3 = peaks_before_drops(x3, t3_d, y3, drops3, wback=180)
    peaks3_sorted = sorted(peaks3, key=lambda p: p[1])
    pL = peaks3_sorted[0]
    pR = peaks3_sorted[-1]
    thr = line_through(pL[1], pL[2], pR[1], pR[2])  # y3 = m*t_days + c

    # --- 2-day overall blue trend (cyan): left edge to right edge ---
    win = 40
    end = len(y2) - 1
    tL = float(np.nanmean(t2_h[0 : min(end, win)]))
    yL = float(np.nanmean(y2[0 : min(end, win)]))
    tR = float(np.nanmean(t2_h[max(0, end - win) : end]))
    yR = float(np.nanmean(y2[max(0, end - win) : end]))
    blue = line_through(tL, yL, tR, yR)  # y2 = m*t_hours + c

    # --- Y-axis translation only (ignore scale differences): pin at most-recent value ---
    # As requested: keep the threshold slope from the 3‑month fit, but translate it to the
    # 2‑day y-axis by matching the right-edge (most recent) blue value.
    y2_now = float(y2[-1])
    y3_now = float(y3[-1])
    delta = y2_now - y3_now  # y2 ≈ y3 + delta

    # --- Draw overlay on 2-day: mapped threshold (red) + blue trend (cyan) ---
    out = im2.convert("RGBA")
    d = ImageDraw.Draw(out, "RGBA")

    # mapped threshold for t in 0..48h (within plot) using t_days = 90 - 2 + t/24 ?
    # The 2-day window corresponds to the last 2 days of the 3-month window.
    # Map 2-day plot time t (0..48h) to 3-month day coordinate: t3 = 90 - 2 + t/24
    def t2_to_t3(t_h):
        return 90.0 - 2.0 + (t_h / 24.0)

    xs = np.linspace(x0, x1, 250)
    t_h = (xs - x0) / (w - 1) * 48.0
    t_d = t2_to_t3(t_h)
    y_thr3 = thr.m * t_d + thr.c
    y_thr2 = y_thr3 + delta

    # clip to crop box
    y_thr2 = np.clip(y_thr2, y0, y1)
    pts_thr = [(float(x), float(y)) for x, y in zip(xs, y_thr2)]
    d.line(pts_thr, fill=(255, 0, 0, 210), width=3)

    # blue trend line segment across full width
    xL = x0
    xR = x1
    tL2 = 0.0
    tR2 = 48.0
    yL2 = blue.y(tL2)
    yR2 = blue.y(tR2)
    yL2 = float(np.clip(yL2, y0, y1))
    yR2 = float(np.clip(yR2, y0, y1))
    d.line([(xL, yL2), (xR, yR2)], fill=(0, 255, 255, 210), width=3)

    # Minimal annotation (no blocking banner)
    d.text(
        (12, 10),
        f"Threshold mapped by y-translation at NOW: delta={delta:.1f} px",
        fill=(255, 255, 255, 235),
    )

    out.save(args.out_two)

    # --- Predict next intersection time (nearest hour), using extrapolation ---
    # Extend both into future: threshold uses thr(t_days), mapped via a,b; blue uses blue(t_hours).
    # Use hours from now h>=0; 2-day plot "now" corresponds to t2=48.
    def y_blue_future(h):
        return blue.y(48.0 + h)

    def y_thr_future(h):
        t_days = 90.0 + h / 24.0
        return (thr.m * t_days + thr.c) + delta

    hs = np.linspace(0, 48, 481)  # next 48h, 6-min resolution
    diff = np.array([y_blue_future(h) - y_thr_future(h) for h in hs], dtype=float)
    cross = None
    for i in range(len(hs) - 1):
        if diff[i] == 0 or diff[i] * diff[i + 1] < 0:
            h0, h1 = hs[i], hs[i + 1]
            d0, d1 = diff[i], diff[i + 1]
            cross = h0 - d0 * (h1 - h0) / (d1 - d0)
            break

    # Draw predicted intersection point on the plot (within next 48h)
    if cross is not None:
        # x position is past the plot (future) if cross>0, so we annotate at right edge with label,
        # and also compute the point at the right edge for reference.
        # For a visual point, clamp to plot range (0..48h) by placing it at t=48 (right edge)
        # and include a note. If cross is <=0, it's inside the plot.
        h_plot = float(np.clip(cross, 0.0, 48.0))
        t_plot = 48.0 if cross >= 0 else 48.0 + h_plot
        # In our parameterization, plot time is 0..48, so use that directly.
        x_int = x0 + ((48.0 if cross > 0 else (48.0 + cross)) / 48.0) * (w - 1)
        y_int = float(y_blue_future(cross))
        y_int = float(np.clip(y_int, y0, y1))
        r = 7
        d.ellipse((x_int - r, y_int - r, x_int + r, y_int + r), outline=(255, 255, 255, 230), width=3)
        d.ellipse((x_int - r + 2, y_int - r + 2, x_int + r - 2, y_int + r - 2), outline=(255, 0, 0, 230), width=2)
        d.text((min(x_int + 10, x1 - 220), max(y_int - 12, y0 + 5)), f"Predicted cross ≈ {cross:.1f}h", fill=(255, 255, 255, 235))
        print(f"hours_to_cross≈{cross:.2f} (nearest hour ≈ {int(round(cross))}h)")
    else:
        print("no crossing found in next 48h")


if __name__ == "__main__":
    raise SystemExit(main())
