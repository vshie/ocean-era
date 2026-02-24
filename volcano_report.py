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
import shutil
import tempfile
import urllib.request

from PIL import Image, ImageOps


PLOTS = {
    "tilt_2day": "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-TILT-2day.png",
    "tilt_week": "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-POC-TILT-week.png",
    "tilt_month": "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-POC-TILT-month.png",
    "tilt_3month": "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-TILT-3month.png",
}


def fetch(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    urllib.request.urlretrieve(url, out_path)


def bundle(panels: list[str], out: str) -> None:
    images = [Image.open(p).convert("RGBA") for p in panels if os.path.exists(p)]
    if not images:
        raise RuntimeError("No panel images were produced")

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
    canvas = Image.new(
        "RGBA",
        (target_w, target_h * len(normalized) + pad * (len(normalized) - 1)),
        (255, 255, 255, 255),
    )
    y = 0
    for panel in normalized:
        canvas.paste(panel, (0, y), panel)
        y += target_h + pad
    canvas.convert("RGB").save(out)


def main() -> int:
    tz = dt.timezone(dt.timedelta(hours=-10))  # HST
    now = dt.datetime.now(tz)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "volcano_report_out")
    os.makedirs(out_dir, exist_ok=True)
    bundle_path = os.path.join(out_dir, "kilauea_predict_bundle.png")

    tmp = tempfile.mkdtemp(prefix="volcano_report_")
    try:
        paths = {}
        for k, url in PLOTS.items():
            p = os.path.join(tmp, f"{k}.png")
            fetch(url, p)
            paths[k] = p

        from volcano_predict_timeseries import main as predict_main

        out_prefix = os.path.join(tmp, "kilauea_predict")
        predict_main([
            "--month", paths["tilt_month"],
            "--three-month", paths["tilt_3month"],
            "--blue-now", "-13.5",
            "--out-prefix", out_prefix,
        ])

        bundle([
            f"{out_prefix}.png",
            f"{out_prefix}_3month_threshold_overlay.png",
            f"{out_prefix}_30day_cyan_overlay.png",
        ], bundle_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nVOLCANO REPORT — Kīlauea")
    print(f"Generated: {now.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"Bundle image: {bundle_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
