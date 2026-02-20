#!/usr/bin/env python3
"""Fetch wind-vs-altitude (pressure levels) for a lat/lon using Open-Meteo,
merge past N days (archive) + next N days (forecast), and generate plots.

Outputs:
  - CSV: long-form rows of time, pressure_hpa, height_m, wind_speed_ms, wind_dir_deg, u_ms, v_ms
  - PNG: time-height heatmap of wind speed + quiver overlay

Example:
  python wind_profile_openmeteo.py --lat 19.4964 --lon -155.4662 --past-days 3 --future-days 3 \
    --tz Pacific/Honolulu --out-prefix hawaii
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
import urllib.parse
import urllib.request


PRESSURE_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]


def iso_date(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")


def http_get_json(url: str, timeout_s: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "clawdbot-wind-profile/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def build_openmeteo_url(
    base: str,
    *,
    lat: float,
    lon: float,
    start_date: str | None = None,
    end_date: str | None = None,
    timezone: str = "UTC",
) -> str:
    # Request wind + direction + geopotential height at several pressure levels.
    hourly_fields = []
    for p in PRESSURE_LEVELS_HPA:
        hourly_fields.extend([
            f"windspeed_{p}hPa",
            f"winddirection_{p}hPa",
            f"geopotential_height_{p}hPa",
        ])

    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "hourly": ",".join(hourly_fields),
        "wind_speed_unit": "mph",  # mph (more intuitive than m/s)
        "timezone": timezone,
    }
    if start_date is not None:
        params["start_date"] = start_date
    if end_date is not None:
        params["end_date"] = end_date

    return base + "?" + urllib.parse.urlencode(params)


def wind_to_uv(speed: float, dir_from_deg: float) -> tuple[float, float]:
    """Convert meteorological direction (degrees FROM which wind blows) to u/v.

    Units: whatever `speed` is in (we request mph). Output u/v in same units.
    u: +east, v: +north
    """
    r = math.radians(dir_from_deg)
    u = -speed * math.sin(r)
    v = -speed * math.cos(r)
    return (u, v)


def parse_dataset(payload: dict) -> list[dict]:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError(f"Unexpected payload (missing hourly/time): keys={list(payload.keys())}")

    times = hourly["time"]

    rows: list[dict] = []
    for i, t in enumerate(times):
        for p in PRESSURE_LEVELS_HPA:
            sp_key = f"windspeed_{p}hPa"
            di_key = f"winddirection_{p}hPa"
            hg_key = f"geopotential_height_{p}hPa"

            sp = hourly.get(sp_key, [None] * len(times))[i]
            di = hourly.get(di_key, [None] * len(times))[i]
            hg = hourly.get(hg_key, [None] * len(times))[i]

            if sp is None or di is None or hg is None:
                continue

            sp_mph = float(sp)
            u, v = wind_to_uv(sp_mph, float(di))
            height_m = float(hg)
            height_ft = height_m * 3.28084
            rows.append(
                {
                    "time": t,
                    "pressure_hpa": p,
                    "height_m": height_m,
                    "height_ft": height_ft,
                    "wind_speed_mph": sp_mph,
                    "wind_dir_deg": float(di),
                    "u_mph": u,
                    "v_mph": v,
                }
            )
    return rows


def merge_rows(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    # Deduplicate by (time, pressure)
    out = {}
    for r in rows_a + rows_b:
        out[(r["time"], r["pressure_hpa"])] = r
    merged = list(out.values())
    merged.sort(key=lambda r: (r["time"], -r["pressure_hpa"]))
    return merged


def rows_to_grids(rows: list[dict]):
    # Unique sorted times
    times = sorted({r["time"] for r in rows})
    levels = sorted({r["pressure_hpa"] for r in rows}, reverse=True)

    # Mean height per level (for plotting as pseudo-altitude axis)
    # Use feet for plotting (more intuitive), but keep meters in the CSV too.
    heights_by_level = {p: [] for p in levels}
    for r in rows:
        heights_by_level[r["pressure_hpa"]].append(r.get("height_ft", r["height_m"] * 3.28084))
    mean_height = {p: sum(v) / len(v) for p, v in heights_by_level.items() if v}

    # Create matrices [L x T]
    speed = [[float("nan") for _ in times] for _ in levels]
    u = [[float("nan") for _ in times] for _ in levels]
    v = [[float("nan") for _ in times] for _ in levels]

    idx_t = {t: j for j, t in enumerate(times)}
    idx_l = {p: i for i, p in enumerate(levels)}

    for r in rows:
        i = idx_l[r["pressure_hpa"]]
        j = idx_t[r["time"]]
        speed[i][j] = r["wind_speed_mph"]
        u[i][j] = r["u_mph"]
        v[i][j] = r["v_mph"]

    y = [mean_height.get(p, float("nan")) for p in levels]
    return times, levels, y, speed, u, v


def plot(times, levels, y_heights_m, speed, u, v, out_png: str, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.dates as mdates
    except Exception as e:
        raise RuntimeError(
            "Plotting requires matplotlib (and numpy). Install them or run without --plot. "
            f"Original error: {e}"
        )

    # Parse times to datetimes for axis formatting
    t_dt = [dt.datetime.fromisoformat(t) for t in times]
    t_num = mdates.date2num(t_dt)

    Z = np.array(speed, dtype=float)

    # Use mean heights as y axis (monotonic increasing with altitude)
    y = np.array(y_heights_m, dtype=float)

    # Sort by height ascending for display
    order = np.argsort(y)
    y2 = y[order]
    Z2 = Z[order, :]
    U2 = np.array(u, dtype=float)[order, :]
    V2 = np.array(v, dtype=float)[order, :]
    levels2 = np.array(levels, dtype=int)[order]

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 1, 1)

    # imshow with extents in data coords
    extent = [t_num[0], t_num[-1], y2[0], y2[-1]]
    im = ax.imshow(
        Z2,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Wind speed (mph)")

    # Quiver overlay, subsample to keep readable
    step_t = max(1, int(len(t_num) / 48))  # ~ every <= 1-3 hours depending length
    step_y = 1 if len(y2) <= 10 else 2

    Tq = t_num[::step_t]
    Yq = y2[::step_y]

    Uq = U2[::step_y, ::step_t]
    Vq = V2[::step_y, ::step_t]

    TT, YY = np.meshgrid(Tq, Yq)

    ax.quiver(
        TT,
        YY,
        Uq,
        Vq,
        color="white",
        alpha=0.7,
        scale=250,
        width=0.0015,
        headwidth=3,
        headlength=4,
    )

    ax.set_title(title)
    ax.set_ylabel("Approx altitude (mean geopotential height, ft)")

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    # Annotate pressure levels on right side (at their mean heights)
    for p, yy in zip(levels2, y2):
        if not np.isfinite(yy):
            continue
        ax.text(t_num[-1], yy, f" {p} hPa", va="center", ha="left", fontsize=8, color="white")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def write_csv(rows: list[dict], out_csv: str):
    fields = [
        "time",
        "pressure_hpa",
        "height_m",
        "height_ft",
        "wind_speed_mph",
        "wind_dir_deg",
        "u_mph",
        "v_mph",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--past-days", type=int, default=3)
    ap.add_argument("--future-days", type=int, default=3)
    ap.add_argument("--tz", type=str, default="Pacific/Honolulu")
    ap.add_argument("--out-prefix", type=str, default="wind_profile")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--title", type=str, default=None, help="Optional plot title override")
    args = ap.parse_args(argv)

    now = dt.datetime.now(dt.timezone.utc).astimezone()  # local tz of runtime
    # Use dates in requested timezone by letting Open-Meteo interpret them via timezone param.
    today = dt.date.today()

    past_start = today - dt.timedelta(days=args.past_days)
    # Open-Meteo archive typically only supports up through *yesterday*.
    # We let the forecast API provide "today" onward.
    past_end = today - dt.timedelta(days=1)

    future_start = today
    future_end = today + dt.timedelta(days=args.future_days)

    # NOTE: Open-Meteo's *archive* endpoint does not reliably provide these
    # pressure-level fields (windspeed_*hPa, etc.) for our use case.
    # The *forecast* endpoint supports `past_days`, which does include pressure levels.
    forecast_base = "https://api.open-meteo.com/v1/forecast"

    url_forecast = build_openmeteo_url(
        forecast_base,
        lat=args.lat,
        lon=args.lon,
        timezone=args.tz,
    )
    # past_days: how many full days before today to include
    # forecast_days: today + N-1 days ahead
    url_forecast += f"&past_days={args.past_days}&forecast_days={args.future_days + 1}"

    payload_f = http_get_json(url_forecast)
    rows = parse_dataset(payload_f)

    out_csv = f"{args.out_prefix}.csv"
    out_png = f"{args.out_prefix}.png"

    write_csv(rows, out_csv)

    if not args.no_plot:
        times, levels, y, speed, u, v = rows_to_grids(rows)
        title = args.title or (
            f"Wind profile (Open-Meteo) @ {args.lat:.4f}, {args.lon:.4f} | "
            f"past {args.past_days}d + next {args.future_days}d"
        )
        plot(times, levels, y, speed, u, v, out_png, title)

    print(f"Wrote: {out_csv}")
    if not args.no_plot:
        print(f"Wrote: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))