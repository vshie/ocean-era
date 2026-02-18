#!/usr/bin/env python3
"""
Generate two buoy drift capture staging points.

Layout (example with northward current):

    L -------- R        (750 ft apart, perpendicular to current)
         |
         |  drift_distance (against current)
         |
         T              (target)

The two points are placed OPPOSITE the current direction from the target,
separated by 750 feet perpendicular to that line.
"""

import argparse
import math
import sys

FEET_PER_METER = 3.28084
METERS_PER_DEGREE_LAT = 111_320  # approximate

CURRENT_BEARINGS = {
    "n":  0,   "north": 0,
    "ne": 45,  "northeast": 45,
    "e":  90,  "east": 90,
    "se": 135, "southeast": 135,
    "s":  180, "south": 180,
    "sw": 225, "southwest": 225,
    "w":  270, "west": 270,
    "nw": 315, "northwest": 315,
}


def offset_point(lat, lon, bearing_deg, distance_m):
    """Offset a lat/lon by a distance (meters) along a bearing (degrees from north CW)."""
    bearing = math.radians(bearing_deg)
    dlat = distance_m * math.cos(bearing) / METERS_PER_DEGREE_LAT
    dlon = distance_m * math.sin(bearing) / (METERS_PER_DEGREE_LAT * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def main():
    parser = argparse.ArgumentParser(
        description="Generate two buoy drift-capture staging points."
    )
    parser.add_argument("-lat", type=float, required=True, help="Target latitude (decimal degrees)")
    parser.add_argument("-lon", type=float, required=True, help="Target longitude (decimal degrees)")
    parser.add_argument("-current", type=str, default="n",
                        help="Current direction the water is flowing TOWARD (e.g. n, ne, south). Default: n")
    parser.add_argument("-drift", type=float, default=1500,
                        help="Drift distance in feet from target to the crossbar (default: 1500)")
    parser.add_argument("-spread", type=float, default=3000,
                        help="Separation between the two staging points in feet (default: 3000)")

    args = parser.parse_args()

    current_key = args.current.strip().lower()
    if current_key not in CURRENT_BEARINGS:
        try:
            current_bearing = float(current_key)
        except ValueError:
            print(f"Error: unknown current direction '{args.current}'")
            print(f"  Use one of: {', '.join(sorted(CURRENT_BEARINGS.keys()))} or a numeric bearing")
            sys.exit(1)
    else:
        current_bearing = CURRENT_BEARINGS[current_key]

    # "Opposite current" means 180 degrees from where current is heading
    against_bearing = (current_bearing + 180) % 360

    drift_m = args.drift / FEET_PER_METER
    half_spread_m = (args.spread / 2) / FEET_PER_METER

    # Center point of the crossbar (straight back from target against current)
    center_lat, center_lon = offset_point(args.lat, args.lon, against_bearing, drift_m)

    # Perpendicular bearings (left and right of the against-current line)
    left_bearing = (against_bearing - 90) % 360
    right_bearing = (against_bearing + 90) % 360

    left_lat, left_lon = offset_point(center_lat, center_lon, left_bearing, half_spread_m)
    right_lat, right_lon = offset_point(center_lat, center_lon, right_bearing, half_spread_m)

    print()
    print("=" * 60)
    print("  BUOY DRIFT CAPTURE - STAGING POINTS")
    print("=" * 60)
    print(f"  Target:           {args.lat:.6f}, {args.lon:.6f}")
    print(f"  Current heading:  {current_bearing}° ({args.current})")
    print(f"  Against current:  {against_bearing}°")
    print(f"  Drift distance:   {args.drift:.0f} ft ({drift_m:.1f} m)")
    print(f"  Point spread:     {args.spread:.0f} ft ({args.spread / FEET_PER_METER:.1f} m)")
    print("-" * 60)
    print()
    print(f"  Left point:   {left_lat:.6f}, {left_lon:.6f}")
    print(f"  Right point:  {right_lat:.6f}, {right_lon:.6f}")
    print()
    print("  --- Copy-paste coordinates ---")
    print(f"  {left_lat:.6f}, {left_lon:.6f}")
    print(f"  {right_lat:.6f}, {right_lon:.6f}")
    print()

    # ASCII diagram
    _draw_diagram(args, current_bearing, left_lat, left_lon, right_lat, right_lon)


def _draw_diagram(args, current_bearing, ll, lo, rl, ro):
    arrow = {
        0: "↑ N", 45: "↗ NE", 90: "→ E", 135: "↘ SE",
        180: "↓ S", 225: "↙ SW", 270: "← W", 315: "↖ NW",
    }
    cur_arrow = arrow.get(int(current_bearing), f"→ {current_bearing}°")

    print("  Layout:")
    print()
    print(f"    L ─────────── R       Current: {cur_arrow}")
    print( "          │")
    print( "          │  drift")
    print( "          │")
    print( "          T (target)")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
