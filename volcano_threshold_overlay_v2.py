#!/usr/bin/env python3
"""Overlay fits for Kilauea tilt plots using PNG-only digitizing.

Requested behavior:
- Use the 3-month plot to fit the drifting threshold trend line.
  Use representative points near the LEFT and RIGHT side of the graph (not a multi-peak regression).
- For the 2-day plot, fit the blue line slope using representative points near the LEFT and RIGHT
  side of the *local ramp* approaching the next drop (not the full 48h).

Outputs are the original PNGs with overlays drawn on top.

Usage:
  python volcano_threshold_overlay_v2.py \
    --two-day volcano/UWD-TILT-2day.png \
    --three-month volcano/UWD-TILT-3month.png \
    --out-three-month volcano/UWD-TILT-3month_overlay.png \
    --out-two volcano/UWD-TILT-2day_overlay_v2.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


CROP = (65, 35, 885, 255)  # tuned for USGS VSC 900x300 images


@dataclass
class Line:
    # y = m*t + c
    m: float
    c: float

    def y(self, t: float) -> float:
        return self.m * t + self.c


def moving_average(a, k=9):
    if k <= 1:
        return a
    k = int(k)
    pad = k // 2
    ap = np.pad(a, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(ap, ker, mode="valid")


def extract_blue_trace(im: Image.Image, crop=CROP):
    rgb = np.array(im.convert("RGB"))
    x0, y0, x1, y1 = crop
    sub = rgb[y0:y1, x0:x1, :]
    r = sub[:, :, 0].astype(np.int16)
    g = sub[:, :, 1].astype(np.int16)
    b = sub[:, :, 2].astype(np.int16)

    # heuristic blue pixel mask
    mask = (b > 130) & (b - r > 40) & (b - g > 30)

    h, w = mask.shape
    ys = np.full(w, np.nan, dtype=float)
    for x in range(w):
        y_idx = np.where(mask[:, x])[0]
        if y_idx.size:
            ys[x] = float(np.median(y_idx))

    xs = np.arange(w, dtype=float) + x0
    ys = ys + y0

    good = np.isfinite(ys)
    if good.sum() >= 2:
        ys[~good] = np.interp(xs[~good], xs[good], ys[good])

    return xs, ys


def line_through(t1, y1, t2, y2) -> Line:
    t1 = float(t1)
    t2 = float(t2)
    y1 = float(y1)
    y2 = float(y2)
    if abs(t2 - t1) < 1e-9:
        return Line(0.0, y1)
    m = (y2 - y1) / (t2 - t1)
    c = y1 - m * t1
    return Line(m, c)


def detect_drops(y, strength=0.6):
    dy = np.diff(y)
    thr = np.nanpercentile(dy, 99) * strength
    idx = np.where(dy > thr)[0]
    drops = []
    last = -999
    for i in idx.tolist():
        if i - last > 10:
            drops.append(i)
            last = i
    return drops


def peaks_before_drops(x, t, y, drops, wback=160):
    peaks = []
    for di in drops:
        lo = max(0, di - wback)
        hi = di
        seg = y[lo:hi]
        if seg.size < 5:
            continue
        j = int(np.nanargmin(seg))  # min y = highest point on plot
        idx = lo + j
        peaks.append((float(x[idx]), float(t[idx]), float(y[idx])))
    return peaks


def draw_three_month_overlay(imm: Image.Image, thr_line: Line, tspan_days: float, peaks, out_path: str):
    im = imm.convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")
    x0, y0, x1, y1 = CROP
    w = x1 - x0

    # draw threshold line across full width
    xs = np.linspace(x0, x1, 250)
    t = (xs - x0) / (w - 1) * tspan_days
    ys = thr_line.m * t + thr_line.c
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
    d.line(pts, fill=(255, 0, 0, 210), width=3)

    # draw the two anchor peaks used
    for (px, pt, py) in peaks:
        r = 6
        d.ellipse((px - r, py - r, px + r, py + r), outline=(255, 0, 0, 230), width=3)

    d.rectangle((10, 10, 560, 42), fill=(0, 0, 0, 140))
    d.text((18, 16), "Overlay: threshold trend (red) anchored by left+right peak points", fill=(255, 255, 255, 235))

    im.save(out_path)


def draw_two_day_overlay(im2: Image.Image, fit_line: Line, tspan_h: float, anchor_pts, out_path: str):
    im = im2.convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")
    x0, y0, x1, y1 = CROP
    w = x1 - x0

    def t_to_x(th):
        return x0 + (th / tspan_h) * (w - 1)

    t1, y1p, t2, y2p = anchor_pts
    x1p = t_to_x(t1)
    x2p = t_to_x(t2)

    # fit line segment (clip to plotting area so it doesn't run off-image)
    x_min, y_min, x_max, y_max = CROP
    y1c = min(max(float(y1p), y_min), y_max)
    y2c = min(max(float(y2p), y_min), y_max)
    d.line([(x1p, y1c), (x2p, y2c)], fill=(0, 255, 255, 210), width=3)

    # mark anchors
    for x, y in [(x1p, y1c), (x2p, y2c)]:
        r = 6
        d.ellipse((x - r, y - r, x + r, y + r), outline=(0, 255, 255, 230), width=3)

    d.rectangle((10, 10, 600, 42), fill=(0, 0, 0, 140))
    d.text((18, 16), "Overlay: local ramp slope (cyan) using left+right points after last drop", fill=(255, 255, 255, 235))

    im.save(out_path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-day", required=True)
    ap.add_argument("--three-month", required=True)
    ap.add_argument("--out-three-month", required=True)
    ap.add_argument("--out-two", required=True)
    args = ap.parse_args(argv)

    im2 = Image.open(args.two_day)
    im3 = Image.open(args.three_month)

    x2, y2 = extract_blue_trace(im2)
    x3, y3 = extract_blue_trace(im3)

    y2s = moving_average(y2, k=9)
    y3s = moving_average(y3, k=11)

    x0, y0, x1, y1 = CROP
    w = x1 - x0

    # Time axes
    t2_h = (x2 - x0) / (w - 1) * 48.0
    t3_d = (x3 - x0) / (w - 1) * 90.0  # treat as 3 months ~ 90 days

    # --- 3-month threshold: detect peaks before drops, then use leftmost+rightmost peaks as anchors ---
    drops3 = detect_drops(y3s)
    peaks3 = peaks_before_drops(x3, t3_d, y3s, drops3, wback=180)

    if len(peaks3) < 2:
        # fallback: take best peak near left and near right by splitting in half
        mid = len(y3s) // 2
        left_idx = int(np.nanargmin(y3s[:mid]))
        right_idx = int(np.nanargmin(y3s[mid:])) + mid
        peaks3 = [(float(x3[left_idx]), float(t3_d[left_idx]), float(y3s[left_idx])), (float(x3[right_idx]), float(t3_d[right_idx]), float(y3s[right_idx]))]

    # choose anchors: smallest t and largest t
    peaks3_sorted = sorted(peaks3, key=lambda p: p[1])
    pL = peaks3_sorted[0]
    pR = peaks3_sorted[-1]
    thr_line = line_through(pL[1], pL[2], pR[1], pR[2])

    # --- 2-day BLUE overall trend: use left-side and right-side anchors across the full plot ---
    # (as requested: left end ~ -2, right end ~ +2 on the plot)
    win = 40
    n = len(y2s)
    end = n - 1

    # pick windows near the left and right edges
    i1 = 0
    i2 = min(end, win)
    j2 = end
    j1 = max(0, end - win)

    t1 = float(np.nanmean(t2_h[i1:i2]))
    y1p = float(np.nanmean(y2s[i1:i2]))

    t2 = float(np.nanmean(t2_h[j1:j2]))
    y2p = float(np.nanmean(y2s[j1:j2]))

    ramp_line = line_through(t1, y1p, t2, y2p)

    draw_three_month_overlay(im3, thr_line, 90.0, [pL, pR], args.out_three_month)
    draw_two_day_overlay(im2, ramp_line, 48.0, (t1, y1p, t2, y2p), args.out_two)

    print("Wrote:", args.out_three_month)
    print("Wrote:", args.out_two)


if __name__ == "__main__":
    raise SystemExit(main())
