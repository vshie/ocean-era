#!/usr/bin/env python3
"""Volcano report (Kīlauea) — fetch USGS plots + compute best-effort next drop intersection.

Outputs:
- Downloads the latest USGS PNGs (tilt plots)
- Computes trendline intersection using PNG digitizing (see volcano_predict_timeseries.py)
- Prints a short text summary and writes assets under ./volcano_report_out/

Intended use: run on demand when user asks for "volcano report".

Notes:
- This is PNG-based digitizing; treat as an estimate.
"""

from __future__ import annotations

import datetime as dt
import os
import urllib.request


PLOTS = {
    "tilt_2day": "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-TILT-2day.png",
    "tilt_week": "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-POC-TILT-week.png",
    "tilt_month": "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-POC-TILT-month.png",
    "tilt_3month": "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-TILT-3month.png",
}


def fetch(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    urllib.request.urlretrieve(url, out_path)


def main() -> int:
    tz = dt.timezone(dt.timedelta(hours=-10))  # HST
    now = dt.datetime.now(tz)

    # Write outputs to repo-root volcano_report_out/, not inside scripts/
    repo_root = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(repo_root, "volcano_report_out")
    os.makedirs(out_dir, exist_ok=True)

    paths = {}
    for k, url in PLOTS.items():
        p = os.path.join(out_dir, f"{k}.png")
        fetch(url, p)
        paths[k] = p

    # Run predictor (imports local module)
    from volcano_predict_timeseries import main as predict_main

    out_prefix = os.path.join(out_dir, "kilauea_predict")
    # blue_now anchored to -13.5 µrad by convention we used with Tony
    import sys
    argv = [
        "--month",
        paths["tilt_month"],
        "--three-month",
        paths["tilt_3month"],
        "--blue-now",
        "-13.5",
        "--out-prefix",
        out_prefix,
    ]
    # call predictor main
    predict_main(argv)

    # Create a single bundled PNG (so we can send one file in chat instead of multiple uploads)
    bundle_path = os.path.join(out_dir, "kilauea_predict_bundle.png")
    try:
        from PIL import Image, ImageOps

        bundle_inputs = [
            f"{out_prefix}.png",
            f"{out_prefix}_3month_threshold_overlay.png",
            f"{out_prefix}_30day_cyan_overlay.png",
        ]
        images = [Image.open(p).convert("RGBA") for p in bundle_inputs if os.path.exists(p)]
        if images:
            # Make every panel the same size (Tony prefers the larger size).
            # Strategy: take the max WxH across all plots, scale each plot to fit,
            # then pad with white to exactly the target size.
            target_w = max(im.width for im in images)
            target_h = max(im.height for im in images)

            normalized = []
            for im in images:
                fitted = ImageOps.contain(im, (target_w, target_h), method=Image.Resampling.LANCZOS)
                panel = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
                x = (target_w - fitted.width) // 2
                y = (target_h - fitted.height) // 2
                panel.paste(fitted, (x, y), fitted)
                normalized.append(panel)

            pad = 12
            out_w = target_w
            out_h = target_h * len(normalized) + pad * (len(normalized) - 1)
            canvas = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))
            y = 0
            for panel in normalized:
                canvas.paste(panel, (0, y), panel)
                y += target_h + pad
            canvas.convert("RGB").save(bundle_path)
    except Exception:
        # Non-fatal: Pillow may not be installed, or a file may be missing.
        bundle_path = ""

    print("\nVOLCANO REPORT — Kīlauea")
    print(f"Generated: {now.strftime('%Y-%m-%d %H:%M %Z')}")
    print("Includes: tilt 2-day, tilt week, tilt month, plus prediction plots")
    print(f"Output dir: {out_dir}")
    if bundle_path:
        print(f"Bundle image: {bundle_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
