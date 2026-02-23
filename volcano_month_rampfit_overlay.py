#!/usr/bin/env python3
"""Overlay a cyan fit on the 30-day (month) tilt plot:
Fit is drawn from the most recent post-drop trough (most recent minimum in data units)
through the most recent data point.

PNG-only digitizing.

Usage:
  python volcano_month_rampfit_overlay.py \
    --month volcano/UWD-POC-TILT-month.png \
    --out volcano/UWD-POC-TILT-month_rampfit_overlay.png
"""

from __future__ import annotations

import argparse
import numpy as np
from PIL import Image, ImageDraw

from volcano_threshold_overlay_v2 import CROP, extract_blue_trace, moving_average, detect_drops


def find_trough_near_x(x, y, *, x_target: float, x_window: float):
    """Find a trough (low in data units => high pixel-y) near a given x position."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = (x >= (x_target - x_window)) & (x <= (x_target + x_window))
    if sel.sum() < 10:
        # fallback: widen
        sel = (x >= (x_target - 2 * x_window)) & (x <= (x_target + 2 * x_window))
    if sel.sum() < 10:
        # fallback: global recent third
        n = len(y)
        lo = int(0.66 * n)
        idx = lo + int(np.nanargmax(y[lo:]))
        return float(x[idx]), float(y[idx])

    yy = y[sel]
    xx = x[sel]
    idx = int(np.nanargmax(yy))
    return float(xx[idx]), float(yy[idx])


def find_now_point(x, y, *, frac_right: float = 0.985):
    """Pick the most recent point near the right edge."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_max = float(np.nanmax(x))
    sel = x >= (x_max * frac_right)
    if sel.sum() < 10:
        sel = x >= (x_max - 20)
    return float(np.nanmean(x[sel])), float(np.nanmean(y[sel]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    im = Image.open(args.month)
    x, y = extract_blue_trace(im)
    y = moving_average(y, 11)

    # Anchor to the *most recent post-drop trough* and the *most recent value at the right edge*.
    # Detect the last sharp "drop" (near-vertical decrease in the plotted quantity).
    x0, y0, x1, y1 = CROP
    w = x1 - x0

    yy = np.asarray(y, dtype=float)
    xx = np.asarray(x, dtype=float)

    dy = np.diff(yy)
    # Sharp drop in data corresponds to a sharp DOWN move on the plot => big +dy in pixel-y.
    thr = np.nanpercentile(dy, 99) * 0.6
    cand = np.where(dy > thr)[0]
    drops = []
    last = -999
    for i in cand.tolist():
        if i - last > 10:
            drops.append(i)
            last = i
    last_drop = drops[-1] if drops else int(0.8 * len(yy))

    # Find trough shortly after the drop: max pixel-y in a window after last_drop.
    # Window size: ~4 days of the 30-day plot.
    px_per_day = w / 30.0
    win_px = int(max(30, 4 * px_per_day))
    lo = min(len(yy) - 2, last_drop + 3)
    hi = min(len(yy), lo + win_px)
    seg = yy[lo:hi]
    if seg.size < 5:
        idx_tr = lo
    else:
        idx_tr = lo + int(np.nanargmax(seg))

    x_tr, y_tr = float(xx[idx_tr]), float(yy[idx_tr])

    # Most recent point: value at the right edge (average of last few samples)
    tail = 20
    x_now = float(np.nanmean(xx[-tail:]))
    y_now = float(np.nanmean(yy[-tail:]))

    # Draw overlay
    out = im.convert("RGBA")
    d = ImageDraw.Draw(out, "RGBA")
    x0, y0, x1, y1 = CROP

    # Cyan fit line segment between trough and now
    d.line([(x_tr, y_tr), (x_now, y_now)], fill=(0, 255, 255, 220), width=3)

    # Mark endpoints
    r = 6
    for xx, yy in [(x_tr, y_tr), (x_now, y_now)]:
        d.ellipse((xx - r, yy - r, xx + r, yy + r), outline=(0, 255, 255, 230), width=3)

    # Estimate slope in grid units (needs horizontal grid spacing in pixels)
    sub = np.array(im.convert('RGB'))[y0:y1, x0:x1]
    rch,gch,bch=sub[:,:,0].astype(int),sub[:,:,1].astype(int),sub[:,:,2].astype(int)
    mx=np.maximum(np.maximum(rch,gch),bch); mn=np.minimum(np.minimum(rch,gch),bch)
    ch=mx-mn; br=(rch+gch+bch)/3
    mask=(ch<20) & (br>120) & (br<230)
    row_counts=mask.sum(axis=1)
    thr=row_counts.mean()+3*row_counts.std()
    rows=np.where(row_counts>thr)[0]
    # cluster and get median spacing
    centers=[]
    if rows.size:
        s=rows[0]; p=rows[0]
        for rr in rows[1:]:
            if rr==p+1:
                p=rr
            else:
                centers.append((s+p)/2)
                s=rr; p=rr
        centers.append((s+p)/2)
    spacing_px=float(np.median(np.diff(centers))) if len(centers)>=2 else float('nan')

    # slope in grid-steps between endpoints
    dy_px = (y_now - y_tr)
    grid_steps = dy_px / spacing_px if np.isfinite(spacing_px) and spacing_px!=0 else float('nan')

    # Small unobtrusive label
    d.text((12, 10), f"Cyan fit: trough(~Jan25) → now(~Jan28). Δy≈{grid_steps:.2f} grid steps", fill=(255, 255, 255, 235))

    out.save(args.out)
    print("Wrote:", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
