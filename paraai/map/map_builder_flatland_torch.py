"""Map builder: flatland map from DEM using PyTorch GPU acceleration.

IMPORTANT: Import this module only AFTER terrain/GDAL has run (e.g. after
RepositoryTerrain.get_elevation). On Windows, importing torch before GDAL
causes access violation (0xC0000005).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from paraai.map.map_builder_base import MapBuilderBase
from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.tool_spacetime import dem_pixel_size_m

logger = logging.getLogger(__name__)


class MapBuilderFlatlandTorch(MapBuilderBase):
    """Build flatland map from DEM using GPU: std and planarity within radius_m."""

    def __init__(
        self,
        radius_m: float = 200.0,
        device: str | torch.device | None = None,
        output_map_names: list[str] | None = None,
    ):
        super().__init__(
            name="MapBuilderFlatlandTorch",
            output_map_names=output_map_names or ["std", "planarity"],
        )
        self.radius_m = radius_m
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        elevation = terrain["elevation"].astype(np.float32)
        elevation = np.nan_to_num(elevation, nan=0.0)
        transform = terrain["transform"]

        elev_t = torch.from_numpy(elevation).to(self.device)
        # Add batch and channel dims: (1, 1, H, W)
        elev_t = elev_t.unsqueeze(0).unsqueeze(0)

        # Std: sqrt(mean(z^2) - mean(z)^2) via avg_pool2d
        kernel = (size_y, size_x)
        padding = (radius_y, radius_x)
        mean_z = F.avg_pool2d(elev_t, kernel, padding=padding)
        mean_z2 = F.avg_pool2d(elev_t**2, kernel, padding=padding)
        std_elev = torch.sqrt(torch.clamp(mean_z2 - mean_z**2, min=0)).squeeze()
        std_elev = std_elev.cpu().numpy().astype(np.float32)

        # Free memory before planarity
        del mean_z, mean_z2
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Planarity: (e2-e3)/(e1-e3) from local 3D covariance, in row batches
        planarity = self._planarity_gpu(elev_t.squeeze(0).squeeze(0), size_y, size_x, radius_y, radius_x)
        planarity = planarity.cpu().numpy().astype(np.float32)
        planarity = np.clip(planarity, 0, 1)

        logger.info(
            "MapBuilderFlatlandTorch: radius=%sm, kernel=%sx%s, device=%s",
            self.radius_m,
            size_y,
            size_x,
            self.device,
        )

        return {
            "std": VectorMapArray(
                "std",
                bounding_box,
                std_elev,
                transform=transform,
            ),
            "planarity": VectorMapArray(
                "planarity",
                bounding_box,
                planarity,
                transform=transform,
            ),
        }

    def _planarity_gpu(
        self,
        elev: torch.Tensor,
        size_y: int,
        size_x: int,
        radius_y: int,
        radius_x: int,
    ) -> torch.Tensor:
        """Vectorized planarity in row batches to limit memory. (e2-e3)/(e1-e3)."""
        h, w = elev.shape
        n = size_y * size_x
        batch_rows = 128

        # Row/col indices for patch (same for all patches) - keep on device, small
        rows = torch.arange(size_y, dtype=elev.dtype, device=elev.device)
        cols = torch.arange(size_x, dtype=elev.dtype, device=elev.device)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        row_col = torch.stack([rr.ravel(), cc.ravel()], dim=1)  # (n, 2)

        planarity_out = torch.zeros(h, w, dtype=elev.dtype, device=elev.device)
        n_batches = (h + batch_rows - 1) // batch_rows

        for batch_start in tqdm(
            range(0, h, batch_rows),
            desc="Planarity (GPU)",
            unit="batch",
            total=n_batches,
            mininterval=1.0,
        ):
            batch_end = min(batch_start + batch_rows, h)
            r0 = max(0, batch_start - radius_y)
            r1 = min(h, batch_end + radius_y)
            chunk = elev[r0:r1, :]  # (chunk_h, w)

            chunk_4d = chunk.unsqueeze(0).unsqueeze(0)
            patches = F.unfold(
                chunk_4d,
                kernel_size=(size_y, size_x),
                padding=(radius_y, radius_x),
            )
            # patches: (1, n, chunk_h*w) -> (chunk_h, w, n)
            chunk_h, _ = chunk.shape
            patches = patches.squeeze(0).T.reshape(chunk_h, w, n)

            # pts: (chunk_h, w, n, 3) - row_col broadcast, elev from patches
            row_col_bc = row_col.unsqueeze(0).unsqueeze(0).expand(chunk_h, w, -1, -1)
            elev_bc = patches.unsqueeze(-1)
            pts = torch.cat([row_col_bc, elev_bc], dim=-1)

            mean_pts = pts.mean(dim=2, keepdim=True)
            pts_centered = pts - mean_pts

            cov = torch.einsum("hwpi,hwpj->hwij", pts_centered, pts_centered) / n
            del pts, pts_centered, patches, row_col_bc, elev_bc

            eigs = torch.linalg.eigvalsh(cov)
            del cov
            e1, e2, e3 = eigs[..., 2], eigs[..., 1], eigs[..., 0]
            del eigs

            denom = e1 - e3
            planarity_batch = torch.where(
                denom > 1e-12,
                (e2 - e3) / denom,
                torch.zeros_like(denom),
            )

            out_r0 = batch_start - r0
            out_r1 = batch_end - r0
            planarity_out[batch_start:batch_end, :] = planarity_batch[out_r0:out_r1, :]

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return planarity_out
