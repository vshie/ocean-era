"""Weather report wrapper for Tony.

Generates wind profiles (Kilauea + Ocean Era), wave report, and conditions
alignment chart, then bundles them into a single weather_report_bundle.png.

Intermediate files are created in a temp directory and cleaned up afterward.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

TZ = "Pacific/Honolulu"
PAST_DAYS = 5
FUTURE_DAYS = 5

KILAUEA = {
    "name": "kilauea",
    "title": "Kilauea",
    "lat": 19.4964,
    "lon": -155.4662,
}

OCEAN_ERA = {
    "name": "oceanera",
    "title": "Ocean Era Site",
    "lat": 19.8290,
    "lon": -156.1208,
}


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def generate_wind(loc: dict, tmp: Path) -> Path:
    script = str(Path(__file__).resolve().parent / "wind_profile_openmeteo.py")
    prefix = tmp / f"wind_{loc['name']}"
    run([
        sys.executable, script,
        "--lat", str(loc["lat"]),
        "--lon", str(loc["lon"]),
        "--past-days", str(PAST_DAYS),
        "--future-days", str(FUTURE_DAYS),
        "--tz", TZ,
        "--title", f"{loc['title']} — Wind Profile ({PAST_DAYS}d past / {FUTURE_DAYS}d forecast)",
        "--out-prefix", str(prefix),
    ])
    return Path(f"{prefix}.png")


def generate_waves(tmp: Path) -> Path:
    script = str(Path(__file__).resolve().parent / "wave_marine_openmeteo.py")
    prefix = tmp / f"waves_{OCEAN_ERA['name']}"
    run([
        sys.executable, script,
        "--lat", str(OCEAN_ERA["lat"]),
        "--lon", str(OCEAN_ERA["lon"]),
        "--tz", TZ,
        "--past-days", str(PAST_DAYS),
        "--future-days", str(FUTURE_DAYS),
        "--title", f"{OCEAN_ERA['title']} — Waves ({PAST_DAYS}d past / {FUTURE_DAYS}d forecast)",
        "--out-prefix", str(prefix),
    ])
    return Path(f"{prefix}.png")


def generate_alignment(tmp: Path) -> Path:
    script = str(Path(__file__).resolve().parent / "ocean_era_alignment.py")
    prefix = tmp / f"alignment_{OCEAN_ERA['name']}"
    run([
        sys.executable, script,
        "--lat", str(OCEAN_ERA["lat"]),
        "--lon", str(OCEAN_ERA["lon"]),
        "--tz", TZ,
        "--days", str(PAST_DAYS + FUTURE_DAYS),
        "--title", f"{OCEAN_ERA['title']} — Conditions + Alignment ({PAST_DAYS}d past / {FUTURE_DAYS}d forecast)",
        "--out-prefix", str(prefix),
    ])
    return Path(f"{prefix}.png")


def bundle(panels: list[Path], out: Path) -> None:
    images = [Image.open(p).convert("RGBA") for p in panels if p.exists()]
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
    out_dir = Path(__file__).resolve().parent / "weather_report_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "weather_report_bundle.png"

    tmp = Path(tempfile.mkdtemp(prefix="weather_report_"))
    try:
        panels = [
            generate_wind(OCEAN_ERA, tmp),
            generate_waves(tmp),
            generate_alignment(tmp),
        ]
        bundle(panels, bundle_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nBundle image: {bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
