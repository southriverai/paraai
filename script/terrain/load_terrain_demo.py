"""Demo script: load terrain for a bounding box and save/summarize output."""

from __future__ import annotations

import argparse
from pathlib import Path

from paraai.tools_terrain import load_terrain


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load terrain elevation and vegetation data for a bounding box."
    )
    parser.add_argument(
        "lon_min",
        type=float,
        help="Minimum longitude (west)",
    )
    parser.add_argument(
        "lat_min",
        type=float,
        help="Minimum latitude (south)",
    )
    parser.add_argument(
        "lon_max",
        type=float,
        help="Maximum longitude (east)",
    )
    parser.add_argument(
        "lat_max",
        type=float,
        help="Maximum latitude (north)",
    )
    parser.add_argument(
        "--dem-resolution",
        type=int,
        default=30,
        choices=[30, 90],
        help="DEM resolution in meters (default: 30)",
    )
    parser.add_argument(
        "--datetime",
        type=str,
        default="2023-01/2023-12",
        help="Sentinel-2 date range (default: 2023-01/2023-12)",
    )
    parser.add_argument(
        "--cloud-cover",
        type=float,
        default=20.0,
        help="Max cloud cover percentage (default: 20)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data", "terrain"),
        help="Cache directory for DEM and imagery (default: data/terrain)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save elevation.tif, rgb.png, ndvi.png (default: none)",
    )
    args = parser.parse_args()

    print(f"Loading terrain for bbox ({args.lon_min}, {args.lat_min}) to ({args.lon_max}, {args.lat_max})...")
    terrain = load_terrain(
        args.lon_min,
        args.lat_min,
        args.lon_max,
        args.lat_max,
        dem_resolution=args.dem_resolution,
        datetime_range=args.datetime,
        cloud_cover_max=args.cloud_cover,
        cache_dir=args.cache_dir,
    )

    elevation = terrain["elevation"]
    rgb = terrain["rgb"]
    ndvi = terrain["ndvi"]

    print(f"  Elevation: shape={elevation.shape}, min={elevation.min():.1f}m, max={elevation.max():.1f}m")
    print(f"  RGB: shape={rgb.shape}")
    print(f"  NDVI: shape={ndvi.shape}, min={ndvi.min():.3f}, max={ndvi.max():.3f}")
    print(f"  CRS: {terrain['crs']}")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        import rasterio

        transform = terrain["transform"]
        crs = terrain["crs"]

        dem_path = args.output_dir / "elevation.tif"
        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            height=elevation.shape[0],
            width=elevation.shape[1],
            count=1,
            dtype=elevation.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(elevation, 1)
        print(f"  Saved {dem_path}")

        rgb_path = args.output_dir / "rgb.png"
        import matplotlib.pyplot as plt

        plt.imsave(rgb_path, rgb)
        print(f"  Saved {rgb_path}")

        ndvi_path = args.output_dir / "ndvi.png"
        ndvi_display = (ndvi + 1) / 2
        plt.imsave(ndvi_path, ndvi_display, cmap="RdYlGn")
        print(f"  Saved {ndvi_path}")


if __name__ == "__main__":
    main()
