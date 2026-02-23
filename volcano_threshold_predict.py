#!/usr/bin/env python3
"""Estimate Kilauea tilt 'dropoff' trigger threshold drift and next drop time from USGS PNG plots.

This is a best-effort image-based digitization (no raw telemetry).
Assumptions:
- UWD-TILT-2day.png spans ~48 hours (left→right).
- UWD-POC-TILT-month.png spans ~30 days (left→right).
- We use the BLUE trace only.

Outputs:
- Predicted time (hours from now) when BLUE trace will intersect the drifting trigger threshold.

Usage:
  python volcano_threshold_predict.py --two-day volcano/UWD-TILT-2day.png --month volcano/UWD-POC-TILT-month.png
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


def _load_rgb(path: str):
    from PIL import Image
    import numpy as np

    im = Image.open(path).convert("RGB")
    return np.array(im)


def _extract_blue_trace(rgb, *, crop):
    """Return x coords and y trace (pixel y, in full-image coordinates) for the blue line."""
    import numpy as np

    x0, y0, x1, y1 = crop
    sub = rgb[y0:y1, x0:x1, :]
    r = sub[:, :, 0].astype(np.int16)
    g = sub[:, :, 1].astype(np.int16)
    b = sub[:, :, 2].astype(np.int16)

    # Heuristic: blue-ish pixels
    mask = (b > 130) & (b - r > 40) & (b - g > 30)

    h, w = mask.shape
    ys = np.full(w, np.nan, dtype=float)

    for x in range(w):
        y_idx = np.where(mask[:, x])[0]
        if y_idx.size == 0:
            continue
        ys[x] = float(np.median(y_idx))

    # convert to full-image coords
    xs = np.arange(w, dtype=float) + x0
    ys_full = ys + y0

    # Fill small gaps by interpolation
    good = np.isfinite(ys_full)
    if good.sum() >= 2:
        ys_full[~good] = np.interp(xs[~good], xs[good], ys_full[good])

    return xs, ys_full


def _moving_average(a, k=9):
    import numpy as np

    if k <= 1:
        return a
    k = int(k)
    pad = k // 2
    ap = np.pad(a, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(ap, ker, mode="valid")


@dataclass
class Fit:
    a: float
    b: float

    def y(self, t: float) -> float:
        return self.a * t + self.b


def _linfit(t, y) -> Fit:
    import numpy as np

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return Fit(float(a), float(b))


def estimate(two_day_png: str, month_png: str):
    import numpy as np

    rgb2 = _load_rgb(two_day_png)
    rgbm = _load_rgb(month_png)

    # Fixed crop tuned for the USGS VSC 900x300 plot layout.
    crop = (65, 35, 885, 255)

    x2, y2 = _extract_blue_trace(rgb2, crop=crop)
    xm, ym = _extract_blue_trace(rgbm, crop=crop)

    # Smooth a bit
    y2s = _moving_average(y2, k=9)
    yms = _moving_average(ym, k=11)

    # Convert x to time axes (in hours/days), assuming full-width spans.
    # We use cropped plot width as the effective time span.
    x0, y0, x1, y1 = crop
    w = x1 - x0

    t2_h = (x2 - x0) / (w - 1) * 48.0  # 0..48 hours (left..right)
    tm_d = (xm - x0) / (w - 1) * 30.0  # 0..30 days (left..right)

    # --- Month plot: find "drop" events and pre-drop peaks ---
    dy = np.diff(yms)
    # A drop is a large positive jump in pixel-y (downwards on screen) across 1 step.
    drop_idx = np.where(dy > np.nanpercentile(dy, 99) * 0.6)[0]

    # Debounce nearby indices
    drops = []
    last = -999
    for i in drop_idx.tolist():
        if i - last > 10:
            drops.append(i)
            last = i

    peaks_t = []
    peaks_y = []
    for di in drops:
        # search window before drop for minimum y (highest point) since y smaller = higher
        wback = 120
        lo = max(0, di - wback)
        hi = di
        seg = yms[lo:hi]
        if seg.size < 5:
            continue
        j = int(np.nanargmin(seg))
        peaks_t.append(tm_d[lo + j])
        peaks_y.append(yms[lo + j])

    # If we didn't find enough drops, fallback to simple local maxima picking
    if len(peaks_t) < 3:
        # crude: take the highest points per 5-day bins
        bins = 6
        for bi in range(bins):
            lo = int(bi * len(yms) / bins)
            hi = int((bi + 1) * len(yms) / bins)
            seg = yms[lo:hi]
            j = int(np.nanargmin(seg))
            peaks_t.append(tm_d[lo + j])
            peaks_y.append(yms[lo + j])

    # Fit drifting threshold in pixel space: y_peak(t_days)
    thr_fit = _linfit(peaks_t, peaks_y)

    # --- Two-day plot: estimate current inflation slope after the last drop ---
    dy2 = np.diff(y2s)
    drop2_idx = np.where(dy2 > np.nanpercentile(dy2, 99) * 0.6)[0]
    last_drop = int(drop2_idx[-1]) if drop2_idx.size else 0

    # Fit slope from shortly after last drop to end (ignore first chunk)
    start = min(len(t2_h) - 5, last_drop + 10)
    t_fit = t2_h[start:]
    y_fit = y2s[start:]
    slope_fit = _linfit(t_fit, y_fit)

    # Current state at right edge (t=48h, t_days=30d)
    t0_h = 48.0
    t0_d = 30.0
    y0 = float(y2s[-1])

    # Solve intersection between y(t) = y0 + s*(t_h - t0_h) and threshold y_thr(t_d)
    # Convert threshold to hours axis near "now" by linearizing thr_fit over time.
    # Since thr_fit is in days, express it as function of hours: t_d = t0_d + (t_h - t0_h)/24.
    # y_thr(t_h) = a*(t0_d + (t_h - t0_h)/24) + b
    a = thr_fit.a
    b = thr_fit.b
    s = slope_fit.a  # pixel-y per hour

    # y0 + s*(t - t0) = a*(t0_d + (t - t0)/24) + b
    # (s - a/24) * (t - t0) = a*t0_d + b - y0
    denom = (s - a / 24.0)
    rhs = (a * t0_d + b - y0)

    if abs(denom) < 1e-9:
        dt_h = float("nan")
    else:
        dt_h = rhs / denom

    # dt_h positive means in the future.
    return {
        "n_peaks": len(peaks_t),
        "threshold_slope_px_per_day": thr_fit.a,
        "two_day_slope_px_per_hour": slope_fit.a,
        "hours_to_cross": float(dt_h),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-day", required=True)
    ap.add_argument("--month", required=True)
    args = ap.parse_args(argv)

    out = estimate(args.two_day, args.month)
    print(out)


if __name__ == "__main__":
    raise SystemExit(main())
