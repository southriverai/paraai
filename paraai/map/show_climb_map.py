"""Shared function to display climb count and strength grids over elevation."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    import pandas as pd

    from paraai.map.vectror_map_array import VectorMapArray

logger = logging.getLogger(__name__)


def show_map_eval(
    vector_map: VectorMapArray,
    holdout_df: pd.DataFrame,
    *,
    column_name: str = "strength",
    title: str | None = None,
    show_random_points: bool = True,
    elevation: np.ndarray | None = None,
) -> None:
    """Scatter plot of map predicted values vs holdout dataframe values, plus vector map."""
    if holdout_df.empty or column_name not in holdout_df.columns:
        return
    plot_df = holdout_df.sample(n=min(10_000, len(holdout_df)), random_state=42) if len(holdout_df) > 10_000 else holdout_df
    pred = vector_map.get_values(plot_df["lat"].to_numpy(), plot_df["lon"].to_numpy())
    actual = plot_df[column_name].to_numpy()

    fig, (ax_map, ax_scatter) = plt.subplots(1, 2, figsize=(14, 6))

    # Vector map: use actual transform extent (DEM may cover larger area than bbox)
    import rasterio.transform as rio_transform

    extent = list(rio_transform.array_bounds(vector_map.array.shape[0], vector_map.array.shape[1], vector_map.transform))
    # array_bounds returns (left, bottom, right, top); imshow extent is [left, right, bottom, top]
    extent = [extent[0], extent[2], extent[1], extent[3]]
    # Heightmap background
    if elevation is not None and elevation.shape == vector_map.array.shape:
        elev_display = np.flipud(
            np.clip(
                (elevation - np.nanpercentile(elevation, 2))
                / (np.nanpercentile(elevation, 98) - np.nanpercentile(elevation, 2) + 1e-10),
                0,
                1,
            )
        )
        ax_map.imshow(elev_display, extent=extent, origin="lower", cmap="gray", interpolation="nearest")
    map_grid = np.flipud(vector_map.array)
    map_masked = np.ma.masked_where(map_grid == 0, map_grid)
    im = ax_map.imshow(map_masked, extent=extent, origin="lower", cmap="coolwarm", interpolation="nearest", alpha=0.7 if elevation is not None else 1.0)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_aspect("equal")
    ax_map.set_title(f"Map: {vector_map.map_name}")
    fig.colorbar(im, ax=ax_map, label=column_name)

    # Scatter: actual vs predicted
    ax_scatter.scatter(actual, pred, alpha=0.3, s=5)
    vmin = min(actual.min(), pred.min())
    vmax = max(actual.max(), pred.max())
    ax_scatter.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, label="Perfect")
    ax_scatter.set_xlabel(f"Actual {column_name}")
    ax_scatter.set_ylabel(f"Predicted {column_name}")
    ax_scatter.set_aspect("equal")
    ax_scatter.legend()
    ax_scatter.set_title("Actual vs predicted")

    if show_random_points and len(plot_df) > 0:
        idx = random.choice(plot_df.index)
        row = plot_df.loc[idx]
        pt_lat, pt_lon = row["lat"], row["lon"]
        pt_actual = row[column_name]
        pt_pred = vector_map.get_values(np.array([pt_lat]), np.array([pt_lon]))[0]
        # Map: mark location
        ax_map.plot(pt_lon, pt_lat, "r.", markersize=10)
        ax_map.axvline(pt_lon, color="r", linestyle=":", alpha=0.7)
        ax_map.axhline(pt_lat, color="r", linestyle=":", alpha=0.7)
        # Scatter: red dotted lines to axes for actual and predicted
        ax_scatter.plot([pt_actual, pt_actual], [vmin, pt_pred], "r:", alpha=0.8)
        ax_scatter.plot([vmin, pt_actual], [pt_pred, pt_pred], "r:", alpha=0.8)
        ax_scatter.plot(pt_actual, pt_pred, "r.", markersize=10)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_climb_map(
    elevation: np.ndarray,
    extent: list[float],
    count_grid: dict[str, np.ndarray],
    strength_grid: np.ndarray,
    *,
    title: str | None = None,
    count_title: str = "Climb count by pixel",
    strength_title: str = "Mean climb strength by pixel",
    save_slippy: str | None = None,
    region_name: str | None = None,
    slippy_max_zoom: int = 11,
    verbose_slippy: bool = True,
) -> None:
    """Plot count and strength grids over elevation background.

    Args:
        elevation: DEM array (rasterio order, row 0 = north).
        extent: [lon_min, lon_max, lat_min, lat_max] for imshow.
        count_grid: Climb count grid, flipped for display (origin lower).
        strength_grid: Mean strength grid, flipped for display (origin lower).
        title: Figure suptitle.
        count_title: Title for count subplot.
        strength_title: Title for strength subplot.
        save_slippy: If set, export to slippy tiles.
        region_name: Used for slippy layer names.
        slippy_max_zoom: Max zoom for slippy tiles.
        verbose_slippy: Verbose slippy tile build.
    """
    elev_display = np.flipud(
        np.clip(
            (elevation - np.nanpercentile(elevation, 2)) / (np.nanpercentile(elevation, 98) - np.nanpercentile(elevation, 2) + 1e-10),
            0,
            1,
        )
    )

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.imshow(elev_display, extent=extent, origin="lower", cmap="gray", interpolation="nearest")
    count_masked = np.ma.masked_where(count_grid == 0, count_grid)
    im1 = ax1.imshow(count_masked, extent=extent, origin="lower", cmap="coolwarm", interpolation="nearest")
    ax1.set_xlim(extent[0], extent[1])
    ax1.set_ylim(extent[2], extent[3])
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_aspect("equal")
    ax1.set_title(count_title)
    fig.colorbar(im1, ax=ax1, label="Climb count")

    ax2.imshow(elev_display, extent=extent, origin="lower", cmap="gray", interpolation="nearest")
    strength_masked = np.ma.masked_where(strength_grid == 0, strength_grid)
    im2 = ax2.imshow(strength_masked, extent=extent, origin="lower", cmap="coolwarm", interpolation="nearest")
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_aspect("equal")
    ax2.set_title(strength_title)
    fig.colorbar(im2, ax=ax2, label="Mean climb strength (m/s)")

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if save_slippy:
        from paraai.tools_terrain import image_to_slippy_tiles

        lon_min_e, lon_max_e, lat_min_e, lat_max_e = extent[0], extent[1], extent[2], extent[3]
        for name, img, cm in [
            ("climb_count", count_grid, "coolwarm"),
            ("climb_strength", strength_grid, "coolwarm"),
        ]:
            layer = f"{save_slippy}_{name}"
            slippy_name = f"{region_name}_{layer}" if region_name else layer
            path = image_to_slippy_tiles(
                img.astype(np.float64),
                lon_min_e,
                lat_min_e,
                lon_max_e,
                lat_max_e,
                slippytilename=slippy_name,
                max_zoom=slippy_max_zoom,
                cmap=cm,
                verbose=verbose_slippy,
            )
            logger.info("Saved slippy tiles: %s", path)

    plt.show()


def show_flatland_map(
    elevation: np.ndarray,
    extent: list[float],
    std_grid: np.ndarray,
    planarity_grid: np.ndarray,
    *,
    title: str | None = None,
    radius_m: float = 200,
) -> None:
    """Plot elevation, std, and planarity grids.

    Args:
        elevation: DEM array (rasterio order, row 0 = north).
        extent: [lon_min, lon_max, lat_min, lat_max] for imshow.
        std_grid: Std elevation grid, flipped for display (origin lower).
        planarity_grid: Planarity grid, flipped for display (origin lower).
        title: Figure suptitle.
        radius_m: Radius in meters (for subplot titles).
    """
    elev_display = np.flipud(
        np.clip(
            (elevation - np.nanpercentile(elevation, 2)) / (np.nanpercentile(elevation, 98) - np.nanpercentile(elevation, 2) + 1e-10),
            0,
            1,
        )
    )

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    ax_elev = fig.add_subplot(gs[0, 0])
    ax_elev.imshow(elev_display, extent=extent, origin="lower", cmap="terrain")
    ax_elev.set_xlabel("Longitude")
    ax_elev.set_ylabel("Latitude")
    ax_elev.set_aspect("equal")
    ax_elev.set_title("Elevation")

    ax_std = fig.add_subplot(gs[0, 1])
    im_std = ax_std.imshow(std_grid, extent=extent, origin="lower", cmap="viridis")
    ax_std.set_xlabel("Longitude")
    ax_std.set_ylabel("Latitude")
    ax_std.set_aspect("equal")
    ax_std.set_title(f"Std elevation ({radius_m}m radius, m)")
    fig.colorbar(im_std, ax=ax_std)

    ax_plan = fig.add_subplot(gs[1, :])
    im_plan = ax_plan.imshow(planarity_grid, extent=extent, origin="lower", cmap="plasma", vmin=0, vmax=1)
    ax_plan.set_xlabel("Longitude")
    ax_plan.set_ylabel("Latitude")
    ax_plan.set_aspect("equal")
    ax_plan.set_title(f"Planarity ({radius_m}m radius)")
    fig.colorbar(im_plan, ax=ax_plan, label="Planarity [0,1]")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
