#!/usr/bin/env python3
"""Generate overlay images for Kilauea tilt plots:
- 30-day plot with estimated drifting threshold line (fit through pre-drop peaks)
- 2-day plot with linear fit of the blue trace after last drop

This uses PNG-only digitizing (no raw telemetry). Output images are the original PNGs
with overlays drawn on top.

Usage:
  python volcano_threshold_overlay.py \
    --two-day volcano/UWD-TILT-2day.png \
    --month volcano/UWD-POC-TILT-month.png \
    --out-month volcano/UWD-POC-TILT-month_overlay.png \
    --out-two volcano/UWD-TILT-2day_overlay.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CROP = (65, 35, 885, 255)  # tuned for USGS VSC 900x300 images


@dataclass
class Fit:
    a: float
    b: float

    def y(self, t: float) -> float:
        return self.a * t + self.b


def linfit(t, y) -> Fit:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return Fit(float(a), float(b))


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


def analyze(two_day_path: str, month_path: str):
    im2 = Image.open(two_day_path)
    imm = Image.open(month_path)

    x2, y2 = extract_blue_trace(im2)
    xm, ym = extract_blue_trace(imm)

    y2s = moving_average(y2, k=9)
    yms = moving_average(ym, k=11)

    x0, y0, x1, y1 = CROP
    w = x1 - x0

    t2_h = (x2 - x0) / (w - 1) * 48.0
    tm_d = (xm - x0) / (w - 1) * 30.0

    # Month drops -> peaks
    dy = np.diff(yms)
    drop_idx = np.where(dy > np.nanpercentile(dy, 99) * 0.6)[0]

    drops = []
    last = -999
    for i in drop_idx.tolist():
        if i - last > 10:
            drops.append(i)
            last = i

    peaks_t, peaks_y, peaks_x = [], [], []
    for di in drops:
        lo = max(0, di - 120)
        hi = di
        seg = yms[lo:hi]
        if seg.size < 5:
            continue
        j = int(np.nanargmin(seg))
        idx = lo + j
        peaks_t.append(tm_d[idx])
        peaks_y.append(yms[idx])
        peaks_x.append(xm[idx])

    thr_fit = linfit(peaks_t, peaks_y)

    # Two-day last drop + slope after
    dy2 = np.diff(y2s)
    drop2_idx = np.where(dy2 > np.nanpercentile(dy2, 99) * 0.6)[0]
    last_drop = int(drop2_idx[-1]) if drop2_idx.size else 0

    start = min(len(t2_h) - 5, last_drop + 10)
    slope_fit = linfit(t2_h[start:], y2s[start:])

    return {
        "im2": im2,
        "imm": imm,
        "x2": x2,
        "y2": y2s,
        "t2_h": t2_h,
        "fit2": slope_fit,
        "start2": start,
        "xm": xm,
        "ym": yms,
        "tm_d": tm_d,
        "fitm": thr_fit,
        "peaks_x": peaks_x,
        "peaks_y": peaks_y,
        "peaks_t": peaks_t,
    }


def draw_month_overlay(imm: Image.Image, fitm: Fit, peaks_x, peaks_y, out_path: str):
    im = imm.convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")
    x0, y0, x1, y1 = CROP

    # Threshold line across full plot width
    w = x1 - x0
    xs = np.linspace(x0, x1, 200)
    # x -> t_days
    t = (xs - x0) / (w - 1) * 30.0
    ys = fitm.a * t + fitm.b
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
    d.line(pts, fill=(255, 0, 0, 200), width=3)

    # Mark peaks
    for x, y in zip(peaks_x, peaks_y):
        r = 5
        d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0, 220), width=2)

    # Label
    d.rectangle((10, 10, 420, 40), fill=(0, 0, 0, 140))
    d.text((18, 16), "Overlay: fitted threshold (red) + detected peaks", fill=(255, 255, 255, 230))

    im.save(out_path)


def draw_two_day_overlay(im2: Image.Image, t2_h, y2, fit2: Fit, start2: int, out_path: str):
    im = im2.convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")
    x0, y0, x1, y1 = CROP
    w = x1 - x0

    # Fit line from start2 to end
    t_start = float(t2_h[start2])
    t_end = float(t2_h[-1])

    def t_to_x(th):
        return x0 + (th / 48.0) * (w - 1)

    x_start = t_to_x(t_start)
    x_end = t_to_x(t_end)
    y_start = fit2.y(t_start)
    y_end = fit2.y(t_end)

    d.line([(x_start, y_start), (x_end, y_end)], fill=(0, 255, 255, 200), width=3)

    # Annotate window
    d.rectangle((10, 10, 520, 40), fill=(0, 0, 0, 140))
    d.text((18, 16), "Overlay: blue-trace linear fit after last drop (cyan)", fill=(255, 255, 255, 230))

    # Mark start point
    r = 5
    d.ellipse((x_start - r, y_start - r, x_start + r, y_start + r), outline=(0, 255, 255, 220), width=2)

    im.save(out_path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-day", required=True)
    ap.add_argument("--month", required=True)
    ap.add_argument("--out-month", required=True)
    ap.add_argument("--out-two", required=True)
    args = ap.parse_args(argv)

    a = analyze(args.two_day, args.month)
    draw_month_overlay(a["imm"], a["fitm"], a["peaks_x"], a["peaks_y"], args.out_month)
    draw_two_day_overlay(a["im2"], a["t2_h"], a["y2"], a["fit2"], a["start2"], args.out_two)

    print("Wrote:", args.out_month)
    print("Wrote:", args.out_two)


if __name__ == "__main__":
    raise SystemExit(main())
