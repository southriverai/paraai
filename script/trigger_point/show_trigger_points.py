from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import contextily as cx
import matplotlib.pyplot as plt

from paraai.repository.repository_trigger_point import RepositoryTriggerPoint
from paraai.tool_spacetime import haversine_km

if TYPE_CHECKING:
    from paraai.model.trigger_point import TriggerPoint


def show_trigger_points(trigger_points: list[TriggerPoint]) -> None:
    """Show trigger points by lat/lon with cartographic basemap."""
    if not trigger_points:
        return
    lats = [tp.lat for tp in trigger_points]
    lons = [tp.lon for tp in trigger_points]
    names = [tp.name for tp in trigger_points]

    fig, ax = plt.subplots(figsize=(10, 8))
    # x=lon, y=lat for correct basemap alignment (EPSG:4326)
    ax.scatter(lons, lats, s=50, alpha=0.7, zorder=5)
    for lat, lon, name in zip(lats, lons, names):
        if not name.lower().startswith("kk"):
            ax.annotate(name, (lon, lat), xytext=(5, 5), textcoords="offset points", fontsize=8, zorder=6)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Trigger points")
    pad_lon = max(0.05, (max(lons) - min(lons)) * 0.1) if lons else 0.05
    pad_lat = max(0.05, (max(lats) - min(lats)) * 0.1) if lats else 0.05
    ax.set_xlim(min(lons) - pad_lon, max(lons) + pad_lon)
    ax.set_ylim(min(lats) - pad_lat, max(lats) + pad_lat)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    with contextlib.suppress(Exception):
        cx.add_basemap(ax, crs="EPSG:4326", alpha=0.6)
    plt.tight_layout()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger_point_id", type=str, help="Trigger point ID")
    parser.add_argument("--name", type=str, help="Trigger point name")
    parser.add_argument("--all", action="store_true", help="Show all trigger points")
    parser.add_argument("radius_km", nargs="?", type=float, help="Show trigger points within N km of the specified point")
    return parser.parse_args()

    # example usage:
    # python script/trigger_point/show_trigger_points.py --name "sopot_house_a" 500
    # python script/trigger_point/show_trigger_points.py --trigger_point_id "sopot_house_a" 500
    # python script/trigger_point/show_trigger_points.py --all


def main() -> None:
    args = parse_args()
    repo_trigger_point = RepositoryTriggerPoint.initialize_sqlite(Path("data", "database_sqlite"))

    # Select trigger point IDs to show
    ids_to_show: list[str] = []
    if args.name:
        center = repo_trigger_point.get_by_name(args.name)
        if center is not None:
            ids_to_show = [center.trigger_point_id]
        else:
            print(f"No trigger point found with name '{args.name}'")
    elif args.trigger_point_id:
        center = repo_trigger_point.get(args.trigger_point_id)
        if center is not None:
            ids_to_show = [args.trigger_point_id]
        else:
            print(f"No trigger point found with id '{args.trigger_point_id}'")
    else:
        ids_to_show = repo_trigger_point.get_all_ids()

    trigger_points = repo_trigger_point.get_by_ids(ids_to_show)

    # Filter by radius if specified (requires --name or --trigger_point_id)
    if args.radius_km is not None:
        if not args.name and not args.trigger_point_id:
            print("radius_km requires --name or --trigger_point_id")
            return
        if not trigger_points:
            return
        center = trigger_points[0]
        all_tps = repo_trigger_point.get_all()
        trigger_points = [
            tp for tp in all_tps
            if haversine_km(center.lat, center.lon, tp.lat, tp.lon) <= args.radius_km
        ]
        print(f"Showing {len(trigger_points)} trigger points within {args.radius_km} km of {center.name}")

    show_trigger_points(trigger_points)
    plt.show()


if __name__ == "__main__":
    main()
