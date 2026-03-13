"""Map builder: flatland map from DEM heightmap (std and planarity within radius)."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy.ndimage import generic_filter, uniform_filter
from tqdm import tqdm

from paraai.map.map_builder_base import MapBuilderBase
from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.tool_spacetime import dem_pixel_size_m

logger = logging.getLogger(__name__)


def _planarity_window(values: np.ndarray, size_y: int, size_x: int) -> float:
    """Planarity from local (row, col, elev) covariance eigenvalues. (e2-e3)/(e1-e3).
    generic_filter passes a 1D array; reshape to (size_y, size_x)."""
    if values.size < 3:
        return 0.0
    vals = values.reshape(size_y, size_x)
    rows = np.arange(size_y, dtype=np.float64).reshape(-1, 1) * np.ones((1, size_x))
    cols = np.ones((size_y, 1)) * np.arange(size_x, dtype=np.float64)
    pts = np.stack([rows.ravel(), cols.ravel(), vals.ravel()], axis=1)
    valid = ~np.isnan(pts[:, 2]) & (np.abs(pts[:, 2]) < 1e6)
    pts = pts[valid]
    if len(pts) < 3:
        return 0.0
    pts = pts - pts.mean(axis=0)
    cov = np.cov(pts.T)
    try:
        eigs = np.linalg.eigvalsh(cov)
        eigs = np.sort(eigs)[::-1]
        e1, e2, e3 = eigs[0], eigs[1], eigs[2]
        denom = e1 - e3
        if denom < 1e-12:
            return 0.0
        return float((e2 - e3) / denom)
    except np.linalg.LinAlgError:
        return 0.0


class MapBuilderFlatland(MapBuilderBase):
    """Build flatland map from DEM: std and planarity of elevation within radius_m."""

    def __init__(
        self,
        radius_m: float = 200.0,
        output_map_names: list[str] | None = None,
    ):
        super().__init__(
            name="MapBuilderFlatland",
            output_map_names=output_map_names or ["std", "planarity"],
        )
        self.radius_m = radius_m

    def get_cache_params(self) -> dict:
        return {"radius_m": self.radius_m}

    def _build_impl(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame,
    ) -> dict[str, VectorMapArray]:
        """Build flatland map from DEM. Returns std (m) and planarity [0,1] within radius_m."""
        _ = df  # unused
        center_lat = (bounding_box.lat_min + bounding_box.lat_max) / 2
        width_m, height_m = dem_pixel_size_m(center_lat)
        radius_x = int(math.ceil(self.radius_m / width_m))
        radius_y = int(math.ceil(self.radius_m / height_m))
        size_x = 2 * radius_x + 1
        size_y = 2 * radius_y + 1

        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)
        elevation = terrain["elevation"].astype(np.float64)
        elevation = np.nan_to_num(elevation, nan=0.0)

        # Std of elevation within radius: sqrt(mean(z^2) - mean(z)^2)
        mean_z = uniform_filter(elevation, size=(size_y, size_x), mode="constant", cval=0)
        mean_z2 = uniform_filter(elevation**2, size=(size_y, size_x), mode="constant", cval=0)
        std_elev = np.sqrt(np.maximum(mean_z2 - mean_z**2, 0)).astype(np.float32)

        # Planarity: (e2-e3)/(e1-e3) from local 3D covariance, in row batches
        def planarity_fn(vals: np.ndarray) -> float:
            return _planarity_window(vals, size_y, size_x)

        h, w = elevation.shape
        batch_rows = 128
        planarity = np.full((h, w), np.nan, dtype=np.float32)
        n_batches = (h + batch_rows - 1) // batch_rows
        logger.info(
            "Planarity: processing %s rows in %s batches (%s rows/batch)",
            h,
            n_batches,
            batch_rows,
        )
        for batch_start in tqdm(
            range(0, h, batch_rows),
            desc="Planarity",
            unit="batch",
            total=n_batches,
            mininterval=1.0,
        ):
            batch_end = min(batch_start + batch_rows, h)
            r0 = max(0, batch_start - radius_y)
            r1 = min(h, batch_end + radius_y)
            chunk = elevation[r0:r1, :]
            result_chunk = generic_filter(
                chunk,
                planarity_fn,
                size=(size_y, size_x),
                mode="constant",
                cval=np.nan,
            )
            out_r0 = batch_start - r0
            out_r1 = batch_end - r0
            planarity[batch_start:batch_end, :] = result_chunk[out_r0:out_r1, :]

        planarity = np.nan_to_num(planarity, nan=0.0)
        planarity = np.clip(planarity, 0, 1)

        logger.info(
            "MapBuilderFlatland: radius=%sm, kernel=%sx%s, planarity done",
            self.radius_m,
            size_y,
            size_x,
        )

        return {
            "std": VectorMapArray(
                "std",
                bounding_box,
                std_elev,
            ),
            "planarity": VectorMapArray(
                "planarity",
                bounding_box,
                planarity,
            ),
        }
