"""Example: load terrain for a specific lat/lon location and plot all bands."""

from pathlib import Path

import matplotlib.pyplot as plt

from paraai.tools_terrain import load_terrain

# # Example: a point near Oslo, Norway
# LAT = 59.91
# LON = 10.75


# Example: a point near Sopot, Bulgaria
LAT = 42.685910
LON = 24.750476

# Bounding box: ~2 km around the point (roughly 0.02 degrees)
LON_MIN = LON - 0.01
LAT_MIN = LAT - 0.01
LON_MAX = LON + 0.01
LAT_MAX = LAT + 0.01

if __name__ == "__main__":
    print(f"Loading terrain for ({LAT}, {LON})...")
    terrain = load_terrain(
        LON_MIN,
        LAT_MIN,
        LON_MAX,
        LAT_MAX,
        cache_dir=Path("data", "terrain"),
    )

    print(f"Elevation: {terrain['elevation'].shape}, range {terrain['elevation'].min():.0f}-{terrain['elevation'].max():.0f} m")
    print(f"NDVI: {terrain['ndvi'].min():.2f} to {terrain['ndvi'].max():.2f}")

    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    axes = axes.flatten()

    axes[0].imshow(terrain["elevation"], cmap="terrain")
    axes[0].set_title("Elevation (m)")
    axes[0].axis("off")

    axes[1].imshow(terrain["red"], cmap="Reds")
    axes[1].set_title("Red (B04)")
    axes[1].axis("off")

    axes[2].imshow(terrain["green"], cmap="Greens")
    axes[2].set_title("Green (B03)")
    axes[2].axis("off")

    axes[3].imshow(terrain["blue"], cmap="Blues")
    axes[3].set_title("Blue (B02)")
    axes[3].axis("off")

    axes[4].imshow(terrain["nir"], cmap="gray")
    axes[4].set_title("NIR (B08)")
    axes[4].axis("off")

    axes[5].imshow(terrain["rgb"])
    axes[5].set_title("RGB")
    axes[5].axis("off")

    axes[6].imshow(terrain["ndvi"], cmap="RdYlGn", vmin=-1, vmax=1)
    axes[6].set_title("NDVI")
    axes[6].axis("off")

    axes[7].axis("off")

    plt.suptitle(f"Terrain data near ({LAT}, {LON})")
    plt.tight_layout()
    plt.show()
