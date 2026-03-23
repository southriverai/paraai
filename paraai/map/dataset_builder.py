"""Map estimate dataset and dataset builder. Extracted from map builders."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from paraai.map.vectror_map_array import VectorMapArray

import pandas as pd
import rasterio
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from paraai.model.boundingbox import BoundingBox
from paraai.model.simple_climb import SimpleClimb
from paraai.repository.repository_datasets import RepositoryDatasets
from paraai.repository.repository_simple_climb import RepositorySimpleClimb
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.tool_string import dict_to_cache_id, validate_safe_name

logger = logging.getLogger(__name__)


def extract_elevation_patch(
    elevation: np.ndarray,
    transform: rasterio.Affine,
    lat: float,
    lon: float,
    radius_m: float,
    size: int,
) -> torch.Tensor:
    """Extract elevation patch centered on (lat, lon), resize to (1, size, size), normalize to [0,1]."""
    bbox = BoundingBox.from_latlon_radius(lat, lon, radius_m)
    lon_min, lat_min, lon_max, lat_max = bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max
    from rasterio.windows import from_bounds

    win = from_bounds(lon_min, lat_min, lon_max, lat_max, transform)
    h, w = elevation.shape
    r0 = int(max(0, win.row_off))
    c0 = int(max(0, win.col_off))
    r1 = int(min(h, win.row_off + win.height))
    c1 = int(min(w, win.col_off + win.width))
    patch = elevation[r0:r1, c0:c1].astype(np.float32)
    valid = ~np.isnan(patch) & (patch > -500)
    if np.any(valid):
        lo, hi = np.percentile(patch[valid], [2, 98])
        patch = np.clip((patch - lo) / (hi - lo + 1e-10), 0, 1)
    else:
        patch = np.zeros_like(patch)
    patch = np.nan_to_num(patch, nan=0.0).astype(np.float32)
    if patch.size == 0:
        return torch.zeros(1, size, size, dtype=torch.float32)
    t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(torch.float32)
    t = nn.functional.interpolate(t, size=(size, size), mode="bilinear").squeeze(0)
    return t


@dataclass
class MapEstimateDatasetResult:
    """Result of building a map estimate dataset."""

    input_maps: list[torch.Tensor]
    target_maps: list[torch.Tensor]
    time_of_day: list[float] | None
    time_of_year: list[float] | None
    ground_alt: list[float]
    start_alt: list[float]
    end_alt: list[float]
    lats: list[float]
    lons: list[float]
    strength_lo: float
    strength_hi: float
    elevation: np.ndarray
    transform: rasterio.Affine


class MapEstimateDataset(Dataset):
    """
    Unified dataset for map estimate models.

    Supports simple (elevation -> target map) and time (elevation + time -> scalar target).
    time_of_day and time_of_year are optional; when present, __getitem__ returns (inp, td, ty, tgt).
    When absent, __getitem__ returns (inp, tgt).
    """

    def __init__(
        self,
        input_maps: list[torch.Tensor],
        target_maps: list[torch.Tensor] | None = None,
        *,
        time_of_day: list[float] | None = None,
        time_of_year: list[float] | None = None,
        example_map: torch.Tensor | None = None,
    ) -> None:
        self.input_maps = input_maps
        if target_maps is not None and len(target_maps) > 0:
            self.target_maps = target_maps
        elif example_map is not None:
            self.target_maps = [example_map] * len(input_maps)
        else:
            raise ValueError("Provide target_maps or example_map")
        self.time_of_day = time_of_day
        self.time_of_year = time_of_year
        assert len(self.input_maps) == len(self.target_maps)
        if time_of_day is not None and time_of_year is not None:
            assert len(self.input_maps) == len(time_of_day) == len(time_of_year)

    def __len__(self) -> int:
        return len(self.input_maps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        inp = self.input_maps[idx]
        tgt = self.target_maps[idx]
        if self.time_of_day is not None and self.time_of_year is not None:
            td = torch.tensor(self.time_of_day[idx], dtype=torch.float32)
            ty = torch.tensor(self.time_of_year[idx], dtype=torch.float32)
            return (inp, td, ty, tgt)
        return (inp, tgt)


class DatasetBuilder:
    """Builds MapEstimateDataset from bounding box and DataFrame(s). Handles cache and get_or_build."""

    def __init__(
        self,
        dataset_builder_type: Literal["climb", "centre", "flat_seed"],
        patch_size_m: float = 500.0,
        image_size: int = 64,
        grid_stride: int = 16,
        test_frac: float = 0.2,
        split_seed: int = 42,
        dataset_limit: int = 300_000,
        *,
        builder_name: str = "MapEstimateDataset",
        flatland_radius_m: float = 200.0,
        flat_seed_planarity_threshold: float = 0.5,
        flat_seed_stride: int = 16,
    ) -> None:
        """Initialize DatasetBuilder with type, patch size, image size, grid stride, test fraction, and split seed.
        For flat_seed: flatland_radius_m, flat_seed_planarity_threshold (planarity > = flatland), flat_seed_stride (pixel stride for sampling).
        dataset_limit: max instances (train+test); excess is randomly sampled down."""

        validate_safe_name(builder_name)
        self.builder_name = builder_name
        self.dataset_builder_type = dataset_builder_type
        self.patch_size_m = patch_size_m
        self.image_size = image_size
        self.grid_stride = grid_stride
        self.test_frac = test_frac
        self.split_seed = split_seed
        self.dataset_limit = dataset_limit
        self.flatland_radius_m = flatland_radius_m
        self.flat_seed_planarity_threshold = flat_seed_planarity_threshold
        self.flat_seed_stride = flat_seed_stride

    def get_dataset_id(self, bounding_boxes: BoundingBox | list[BoundingBox]) -> str:
        """Params for dataset cache key (includes split params and dataset_builder_type)."""
        bbox_list = bounding_boxes if isinstance(bounding_boxes, list) else [bounding_boxes]
        params: dict = dict(
            builder_name=self.builder_name,
            dataset_builder_type=self.dataset_builder_type,
            patch_size_m=self.patch_size_m,
            image_size=self.image_size,
            grid_stride=self.grid_stride,
            test_frac=self.test_frac,
            split_seed=self.split_seed,
            bboxes=[[b.lat_min, b.lat_max, b.lon_min, b.lon_max] for b in bbox_list],
        )
        if self.dataset_builder_type == "flat_seed":
            params["flatland_radius_m"] = self.flatland_radius_m
            params["flat_seed_planarity_threshold"] = self.flat_seed_planarity_threshold
            params["flat_seed_stride"] = self.flat_seed_stride
        params["dataset_limit"] = self.dataset_limit
        return dict_to_cache_id(**params)

    @staticmethod
    def _alt_norm(alt_m: float) -> float:
        """Rescale altitude to 0-1 by dividing by 10000."""
        return float(np.clip(alt_m / 10000.0, 0.0, 1.0))

    @staticmethod
    def _climb_row(c: SimpleClimb) -> dict:
        """Single climb as dict row for DataFrame."""
        return {
            "ground_lat": c.ground_lat,
            "ground_lon": c.ground_lon,
            "climb_strength_m_s": c.climb_strength_m_s,
            "time_of_day_h": c.time_of_day_h,
            "time_of_year_d": c.time_of_year_d,
            "ground_alt_norm": DatasetBuilder._alt_norm(c.ground_alt_m),
            "start_alt_norm": DatasetBuilder._alt_norm(c.start_alt_m),
            "end_alt_norm": DatasetBuilder._alt_norm(c.end_alt_m),
        }

    def _expand_climbs_mode_climb(self, climbs: list[SimpleClimb]) -> pd.DataFrame:
        """One point per climb."""
        return pd.DataFrame([self._climb_row(c) for c in climbs])

    def _expand_climbs_mode_centre(self, climbs: list[SimpleClimb]) -> pd.DataFrame:
        """Group by tracklog_id; between each pair of consecutive climbs add synthetic zero-climb point."""
        by_tracklog: dict[str, list[SimpleClimb]] = defaultdict(list)
        for c in climbs:
            by_tracklog[c.tracklog_id].append(c)

        rows: list[dict] = []
        for tracklog_climbs in by_tracklog.values():
            sorted_climbs = sorted(tracklog_climbs, key=lambda x: x.start_timestamp_utc)
            if len(sorted_climbs) < 2:
                rows.extend(self._climb_row(c) for c in sorted_climbs)
                continue

            ground_centre_lat = sum(c.ground_lat for c in sorted_climbs) / len(sorted_climbs)
            ground_centre_lon = sum(c.ground_lon for c in sorted_climbs) / len(sorted_climbs)

            for i, c in enumerate(sorted_climbs):
                rows.append(self._climb_row(c))
                if i < len(sorted_climbs) - 1:
                    next_c = sorted_climbs[i + 1]
                    mid_end_start_lat = (c.end_lat + next_c.start_lat) / 2.0
                    mid_end_start_lon = (c.end_lon + next_c.start_lon) / 2.0
                    synth_lat = (ground_centre_lat + mid_end_start_lat) / 2.0
                    synth_lon = (ground_centre_lon + mid_end_start_lon) / 2.0
                    synth_td = (c.time_of_day_h + next_c.time_of_day_h) / 2.0
                    synth_ty = (c.time_of_year_d + next_c.time_of_year_d) / 2.0
                    synth_ground_alt = DatasetBuilder._alt_norm((c.ground_alt_m + next_c.ground_alt_m) / 2.0)
                    synth_start_alt = DatasetBuilder._alt_norm((c.start_alt_m + next_c.start_alt_m) / 2.0)
                    synth_end_alt = DatasetBuilder._alt_norm((c.end_alt_m + next_c.end_alt_m) / 2.0)
                    rows.append(
                        {
                            "ground_lat": synth_lat,
                            "ground_lon": synth_lon,
                            "climb_strength_m_s": 0.0,
                            "time_of_day_h": synth_td,
                            "time_of_year_d": synth_ty,
                            "ground_alt_norm": synth_ground_alt,
                            "start_alt_norm": synth_start_alt,
                            "end_alt_norm": synth_end_alt,
                        }
                    )

        return pd.DataFrame(rows)

    def _expand_climbs_to_points(self, climbs: list[SimpleClimb]) -> pd.DataFrame:
        """Expand climbs to dataset points based on mode."""
        if self.dataset_builder_type == "climb":
            return self._expand_climbs_mode_climb(climbs)
        if self.dataset_builder_type == "centre":
            return self._expand_climbs_mode_centre(climbs)
        if self.dataset_builder_type == "flat_seed":
            return self._expand_climbs_mode_climb(climbs)
        raise ValueError(f"Unknown dataset_builder_type: {self.dataset_builder_type!r}")

    def _sample_flatland_points(
        self,
        planarity_vma: VectorMapArray,
        elevation: np.ndarray,
        transform: rasterio.Affine,
        strength_lo: float,
        strength_hi: float,
    ) -> pd.DataFrame:
        """Sample 0-climb points from flatland pixels (planarity > threshold)."""
        planarity = planarity_vma.array
        h, w = planarity.shape
        stride = self.flat_seed_stride
        thresh = self.flat_seed_planarity_threshold

        rows: list[dict] = []
        for r in range(0, h, stride):
            for c in range(0, w, stride):
                if planarity[r, c] <= thresh:
                    continue
                lon, lat = rasterio.transform.xy(transform, r, c)
                lat, lon = float(lat), float(lon)
                elev_val = float(np.nan_to_num(elevation[r, c], nan=0.0))
                alt_norm = self._alt_norm(elev_val)
                rows.append(
                    {
                        "ground_lat": lat,
                        "ground_lon": lon,
                        "climb_strength_m_s": 0.0,
                        "time_of_day_h": 12.0,
                        "time_of_year_d": 182.5,
                        "ground_alt_norm": alt_norm,
                        "start_alt_norm": alt_norm,
                        "end_alt_norm": alt_norm,
                    }
                )
        return pd.DataFrame(rows)

    def split_climbs(
        self,
        climbs: list[SimpleClimb],
    ) -> tuple[list[SimpleClimb], list[SimpleClimb]]:
        """Split climbs into train and test by tracklog_id. Entire tracklogs go to train or test."""
        by_tracklog: dict[str, list[SimpleClimb]] = defaultdict(list)
        for c in climbs:
            by_tracklog[c.tracklog_id].append(c)

        tracklog_ids = list(by_tracklog.keys())
        random.Random(self.split_seed).shuffle(tracklog_ids)

        n_test_desired = max(1, int(len(climbs) * self.test_frac))

        train_climbs: list[SimpleClimb] = []
        test_climbs: list[SimpleClimb] = []
        n_test_so_far = 0
        for tid in tracklog_ids:
            tracklog_climbs = by_tracklog[tid]
            if n_test_so_far < n_test_desired:
                test_climbs.extend(tracklog_climbs)
                n_test_so_far += len(tracklog_climbs)
            else:
                train_climbs.extend(tracklog_climbs)

        n_train_actual = len(train_climbs)
        n_test_actual = len(test_climbs)
        train_frac_desired = 1.0 - self.test_frac
        test_frac_desired = self.test_frac
        train_frac_actual = n_train_actual / len(climbs) if climbs else 0.0
        test_frac_actual = n_test_actual / len(climbs) if climbs else 0.0
        logger.info(
            "Split by tracklog: desired train=%.1f%% test=%.1f%% | actual train=%d (%.1f%%) test=%d (%.1f%%) | %d tracklogs",
            train_frac_desired * 100,
            test_frac_desired * 100,
            n_train_actual,
            train_frac_actual * 100,
            n_test_actual,
            test_frac_actual * 100,
            len(tracklog_ids),
        )
        return train_climbs, test_climbs

    def build(
        self,
        bounding_box: BoundingBox,
        climbs_train: list[SimpleClimb],
        climbs_test: list[SimpleClimb],
    ) -> tuple[MapEstimateDatasetResult, MapEstimateDatasetResult, int]:
        """
        Build train and test dataset results from SimpleClimbs.
        Always uses time (time_of_day, time_of_year) and scalar targets.
        dataset_mode: "climb" (one point per climb) or "centre" (add zero-climb points between climbs).
        Returns (train_result, test_result, n_zero_climb_total).
        """
        df_train = self._expand_climbs_to_points(climbs_train)
        df_test = self._expand_climbs_to_points(climbs_test)
        n_zero_climb_total = int((df_train["climb_strength_m_s"] == 0).sum() + (df_test["climb_strength_m_s"] == 0).sum())

        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)
        elevation = terrain["elevation"]
        transform = terrain["transform"]

        strength_vals = df_train["climb_strength_m_s"].to_numpy(dtype=np.float64)
        valid = strength_vals[~np.isnan(strength_vals) & (strength_vals > 0)]
        if valid.size > 0:
            p2, p98 = np.percentile(valid, [2, 98])
            strength_lo, strength_hi = float(p2), float(p98)
        else:
            strength_lo, strength_hi = 0.0, 1.0
        if strength_hi <= strength_lo:
            strength_hi = strength_lo + 1e-6

        train_result = self._build_from_points(df_train, elevation, transform, strength_lo, strength_hi, "Train patches")
        test_result = self._build_from_points(df_test, elevation, transform, strength_lo, strength_hi, "Test patches")

        return train_result, test_result, n_zero_climb_total

    def _build_from_multiple_bboxes(
        self,
        bounding_boxes: list[BoundingBox],
        climbs_train: list[SimpleClimb],
        climbs_test: list[SimpleClimb],
    ) -> tuple[MapEstimateDatasetResult, MapEstimateDatasetResult, int]:
        """Build train/test results using per-bbox terrain (no union). Points are grouped by containing bbox."""
        df_train = self._expand_climbs_to_points(climbs_train)
        df_test = self._expand_climbs_to_points(climbs_test)
        n_zero_climb_total = int((df_train["climb_strength_m_s"] == 0).sum() + (df_test["climb_strength_m_s"] == 0).sum())

        strength_vals = df_train["climb_strength_m_s"].to_numpy(dtype=np.float64)
        valid = strength_vals[~np.isnan(strength_vals) & (strength_vals > 0)]
        if valid.size > 0:
            p2, p98 = np.percentile(valid, [2, 98])
            strength_lo, strength_hi = float(p2), float(p98)
        else:
            strength_lo, strength_hi = 0.0, 1.0
        if strength_hi <= strength_lo:
            strength_hi = strength_lo + 1e-6

        def _points_in_bbox(df: pd.DataFrame, bbox: BoundingBox) -> pd.DataFrame:
            mask = df.apply(lambda r: bbox.is_in(r["ground_lat"], r["ground_lon"]), axis=1)
            return df[mask]

        def _merge_results(a: MapEstimateDatasetResult | None, b: MapEstimateDatasetResult) -> MapEstimateDatasetResult:
            if a is None or len(a.input_maps) == 0:
                return b
            return MapEstimateDatasetResult(
                input_maps=a.input_maps + b.input_maps,
                target_maps=a.target_maps + b.target_maps,
                time_of_day=(a.time_of_day or []) + (b.time_of_day or []),
                time_of_year=(a.time_of_year or []) + (b.time_of_year or []),
                ground_alt=a.ground_alt + b.ground_alt,
                start_alt=a.start_alt + b.start_alt,
                end_alt=a.end_alt + b.end_alt,
                lats=a.lats + b.lats,
                lons=a.lons + b.lons,
                strength_lo=a.strength_lo,
                strength_hi=a.strength_hi,
                elevation=b.elevation,
                transform=b.transform,
            )

        train_result: MapEstimateDatasetResult | None = None
        test_result: MapEstimateDatasetResult | None = None
        repo_terrain = RepositoryTerrain.get_instance()

        for bbox in bounding_boxes:
            train_df_b = _points_in_bbox(df_train, bbox)
            test_df_b = _points_in_bbox(df_test, bbox)
            if len(train_df_b) == 0 and len(test_df_b) == 0:
                continue
            terrain = repo_terrain.get_elevation(bbox)
            elevation = terrain["elevation"]
            transform = terrain["transform"]
            if len(train_df_b) > 0:
                tr_b = self._build_from_points(
                    train_df_b, elevation, transform, strength_lo, strength_hi, f"Train patches ({bbox})"
                )
                train_result = _merge_results(train_result, tr_b)
            if len(test_df_b) > 0:
                te_b = self._build_from_points(
                    test_df_b, elevation, transform, strength_lo, strength_hi, f"Test patches ({bbox})"
                )
                test_result = _merge_results(test_result, te_b)

        if train_result is None or test_result is None:
            raise ValueError("No climbs in any bounding box")
        return train_result, test_result, n_zero_climb_total

    @staticmethod
    def _subsample_result(result: MapEstimateDatasetResult, indices: list[int]) -> MapEstimateDatasetResult:
        """Return new MapEstimateDatasetResult with only the given indices."""
        return MapEstimateDatasetResult(
            input_maps=[result.input_maps[i] for i in indices],
            target_maps=[result.target_maps[i] for i in indices],
            time_of_day=[result.time_of_day[i] for i in indices] if result.time_of_day else None,
            time_of_year=[result.time_of_year[i] for i in indices] if result.time_of_year else None,
            ground_alt=[result.ground_alt[i] for i in indices],
            start_alt=[result.start_alt[i] for i in indices],
            end_alt=[result.end_alt[i] for i in indices],
            lats=[result.lats[i] for i in indices],
            lons=[result.lons[i] for i in indices],
            strength_lo=result.strength_lo,
            strength_hi=result.strength_hi,
            elevation=result.elevation,
            transform=result.transform,
        )

    def _apply_dataset_limit(
        self,
        train_result: MapEstimateDatasetResult,
        test_result: MapEstimateDatasetResult,
    ) -> tuple[MapEstimateDatasetResult, MapEstimateDatasetResult]:
        """Randomly subsample train+test to dataset_limit if total exceeds it."""
        n_train = len(train_result.input_maps)
        n_test = len(test_result.input_maps)
        total = n_train + n_test
        if total <= self.dataset_limit:
            return train_result, test_result
        rng = random.Random(self.split_seed)
        n_train_new = int(self.dataset_limit * n_train / total)
        n_test_new = self.dataset_limit - n_train_new
        if n_test_new <= 0:
            n_test_new = 1
            n_train_new = self.dataset_limit - 1
        train_idx = sorted(rng.sample(range(n_train), n_train_new))
        test_idx = sorted(rng.sample(range(n_test), n_test_new))
        logger.info(
            "Dataset limit %d: randomly sampled from %d to %d train, %d to %d test",
            self.dataset_limit, n_train, n_train_new, n_test, n_test_new,
        )
        return (
            self._subsample_result(train_result, train_idx),
            self._subsample_result(test_result, test_idx),
        )

    def _build_from_points(
        self,
        df: pd.DataFrame,
        elevation: np.ndarray,
        transform: rasterio.Affine,
        strength_lo: float,
        strength_hi: float,
        desc: str,
    ) -> MapEstimateDatasetResult:
        inputs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        time_day: list[float] = []
        time_year: list[float] = []
        ground_alt: list[float] = []
        start_alt: list[float] = []
        end_alt: list[float] = []
        lats: list[float] = []
        lons: list[float] = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc, unit="patch"):
            lat_val = row["ground_lat"]
            lon_val = row["ground_lon"]
            raw_val = row["climb_strength_m_s"]
            patch = extract_elevation_patch(elevation, transform, lat_val, lon_val, self.patch_size_m, self.image_size)
            norm_val = float(np.clip((raw_val - strength_lo) / (strength_hi - strength_lo), 0, 1))

            inputs.append(patch)
            lats.append(lat_val)
            lons.append(lon_val)
            time_day.append(float(np.clip(row["time_of_day_h"] / 24.0, 0.0, 1.0)))
            time_year.append(float(np.clip(row["time_of_year_d"] / 365.0, 0.0, 1.0)))
            ground_alt.append(float(row["ground_alt_norm"]))
            start_alt.append(float(row["start_alt_norm"]))
            end_alt.append(float(row["end_alt_norm"]))
            targets.append(torch.tensor([norm_val], dtype=torch.float32))

        return MapEstimateDatasetResult(
            input_maps=inputs,
            target_maps=targets,
            time_of_day=time_day,
            time_of_year=time_year,
            ground_alt=ground_alt,
            start_alt=start_alt,
            end_alt=end_alt,
            lats=lats,
            lons=lons,
            strength_lo=strength_lo,
            strength_hi=strength_hi,
            elevation=elevation,
            transform=transform,
        )

    def _build_dataset_flat_seed_multi(
        self,
        bounding_boxes: list[BoundingBox],
        climbs: list[SimpleClimb],
    ) -> tuple[MapEstimateDatasetResult, MapEstimateDatasetResult, int]:
        """Build flat_seed dataset for multiple bboxes: filter flatlands per bbox, build with per-bbox terrain."""
        from paraai.map.map_builder_flatland_torch import MapBuilderFlatlandTorch

        builder_flatland = MapBuilderFlatlandTorch(radius_m=self.flatland_radius_m)
        thresh = self.flat_seed_planarity_threshold
        climbs_filtered: list[SimpleClimb] = []
        for bbox in bounding_boxes:
            climbs_in_bbox = [c for c in climbs if bbox.is_in(c.ground_lat, c.ground_lon)]
            if not climbs_in_bbox:
                continue
            flatland_maps = builder_flatland.build(bbox, None)
            planarity_vma = flatland_maps["planarity"]
            lats = [c.ground_lat for c in climbs_in_bbox]
            lons = [c.ground_lon for c in climbs_in_bbox]
            planarity_at_climbs = planarity_vma.get_values(lats, lons)
            for i, c in enumerate(climbs_in_bbox):
                if planarity_at_climbs[i] <= thresh:
                    climbs_filtered.append(c)
        climbs_filtered = list({c.simple_climb_id: c for c in climbs_filtered}.values())
        removed = len(climbs) - len(climbs_filtered)
        if removed > 0:
            logger.info(
                "flat_seed: removed %d climbs in flatlands (planarity > %.2f), keeping %d",
                removed,
                thresh,
                len(climbs_filtered),
            )
        if len(climbs_filtered) < 2:
            raise ValueError(
                f"flat_seed: need at least 2 climbs after filtering flatlands, got {len(climbs_filtered)}. "
                "Try lowering flat_seed_planarity_threshold."
            )
        train_climbs, test_climbs = self.split_climbs(climbs_filtered)
        train_result, test_result, n_zero_climb = self._build_from_multiple_bboxes(
            bounding_boxes, train_climbs, test_climbs
        )
        repo_terrain = RepositoryTerrain.get_instance()
        for bbox in bounding_boxes:
            flatland_maps = builder_flatland.build(bbox, None)
            planarity_vma = flatland_maps["planarity"]
            terrain = repo_terrain.get_elevation(bbox)
            elevation = terrain["elevation"]
            transform = terrain["transform"]
            flatland_df = self._sample_flatland_points(
                planarity_vma, elevation, transform,
                train_result.strength_lo, train_result.strength_hi,
            )
            if len(flatland_df) == 0:
                continue
            indices = np.arange(len(flatland_df))
            rng = random.Random(self.split_seed)
            rng.shuffle(indices)
            n_test_flat = max(0, int(len(flatland_df) * self.test_frac))
            n_train_flat = len(flatland_df) - n_test_flat
            train_flat_df = flatland_df.iloc[indices[:n_train_flat]]
            test_flat_df = flatland_df.iloc[indices[n_train_flat:]]
            flat_train = self._build_from_points(
                train_flat_df, elevation, transform,
                train_result.strength_lo, train_result.strength_hi, "Flatland train",
            )
            flat_test = self._build_from_points(
                test_flat_df, elevation, transform,
                train_result.strength_lo, train_result.strength_hi, "Flatland test",
            )

            def _merge(a: MapEstimateDatasetResult, b: MapEstimateDatasetResult) -> MapEstimateDatasetResult:
                return MapEstimateDatasetResult(
                    input_maps=a.input_maps + b.input_maps,
                    target_maps=a.target_maps + b.target_maps,
                    time_of_day=(a.time_of_day or []) + (b.time_of_day or []),
                    time_of_year=(a.time_of_year or []) + (b.time_of_year or []),
                    ground_alt=a.ground_alt + b.ground_alt,
                    start_alt=a.start_alt + b.start_alt,
                    end_alt=a.end_alt + b.end_alt,
                    lats=a.lats + b.lats,
                    lons=a.lons + b.lons,
                    strength_lo=a.strength_lo,
                    strength_hi=a.strength_hi,
                    elevation=b.elevation,
                    transform=b.transform,
                )

            train_result = _merge(train_result, flat_train)
            test_result = _merge(test_result, flat_test)
            n_zero_climb += len(flatland_df)
            logger.info(
                "flat_seed: added %d flatland 0-climb points from bbox (%d train, %d test)",
                len(flatland_df), n_train_flat, n_test_flat,
            )
        return train_result, test_result, n_zero_climb

    def _build_dataset_flat_seed(
        self,
        bounding_box: BoundingBox,
        climbs: list[SimpleClimb],
    ) -> tuple[MapEstimateDatasetResult, MapEstimateDatasetResult, int]:
        """Build dataset for flat_seed: remove climbs in flatlands, seed 0-climb points from flatlands."""
        from paraai.map.map_builder_flatland_torch import MapBuilderFlatlandTorch

        builder_flatland = MapBuilderFlatlandTorch(radius_m=self.flatland_radius_m)
        flatland_maps = builder_flatland.build(bounding_box, None)
        planarity_vma = flatland_maps["planarity"]

        lats = [c.ground_lat for c in climbs]
        lons = [c.ground_lon for c in climbs]
        planarity_at_climbs = planarity_vma.get_values(lats, lons)
        thresh = self.flat_seed_planarity_threshold
        climbs_filtered = [c for i, c in enumerate(climbs) if planarity_at_climbs[i] <= thresh]
        removed = len(climbs) - len(climbs_filtered)
        if removed > 0:
            logger.info(
                "flat_seed: removed %d climbs in flatlands (planarity > %.2f), keeping %d",
                removed,
                thresh,
                len(climbs_filtered),
            )
        if len(climbs_filtered) < 2:
            raise ValueError(
                f"flat_seed: need at least 2 climbs after filtering flatlands, got {len(climbs_filtered)}. "
                "Try lowering flat_seed_planarity_threshold."
            )

        train_climbs, test_climbs = self.split_climbs(climbs_filtered)
        train_result, test_result, n_zero_climb = self.build(bounding_box, train_climbs, test_climbs)

        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)
        elevation = terrain["elevation"]
        transform = terrain["transform"]

        flatland_df = self._sample_flatland_points(
            planarity_vma, elevation, transform, train_result.strength_lo, train_result.strength_hi
        )
        if len(flatland_df) > 0:
            indices = np.arange(len(flatland_df))
            rng = random.Random(self.split_seed)
            rng.shuffle(indices)
            n_test_flat = max(0, int(len(flatland_df) * self.test_frac))
            n_train_flat = len(flatland_df) - n_test_flat
            train_flat_df = flatland_df.iloc[indices[:n_train_flat]]
            test_flat_df = flatland_df.iloc[indices[n_train_flat:]]

            flat_train = self._build_from_points(
                train_flat_df, elevation, transform,
                train_result.strength_lo, train_result.strength_hi, "Flatland train",
            )
            flat_test = self._build_from_points(
                test_flat_df, elevation, transform,
                train_result.strength_lo, train_result.strength_hi, "Flatland test",
            )

            def _merge(a: MapEstimateDatasetResult, b: MapEstimateDatasetResult) -> MapEstimateDatasetResult:
                return MapEstimateDatasetResult(
                    input_maps=a.input_maps + b.input_maps,
                    target_maps=a.target_maps + b.target_maps,
                    time_of_day=a.time_of_day + b.time_of_day,
                    time_of_year=a.time_of_year + b.time_of_year,
                    ground_alt=a.ground_alt + b.ground_alt,
                    start_alt=a.start_alt + b.start_alt,
                    end_alt=a.end_alt + b.end_alt,
                    lats=a.lats + b.lats,
                    lons=a.lons + b.lons,
                    strength_lo=a.strength_lo,
                    strength_hi=a.strength_hi,
                    elevation=a.elevation,
                    transform=a.transform,
                )

            train_result = _merge(train_result, flat_train)
            test_result = _merge(test_result, flat_test)
            n_zero_climb += len(flatland_df)
            logger.info(
                "flat_seed: added %d flatland 0-climb points (%d train, %d test)",
                len(flatland_df), n_train_flat, n_test_flat,
            )

        return train_result, test_result, n_zero_climb

    async def build_dataset(
        self,
        bounding_boxes: BoundingBox | list[BoundingBox],
        *,
        ignore_cache: bool = False,
    ) -> dict:
        """
        Build dataset with cache support from SimpleClimbs.
        For multiple bounding boxes: fetches climbs from each bbox separately (no union), merges, builds with per-bbox terrain.
        """
        bbox_list = bounding_boxes if isinstance(bounding_boxes, list) else [bounding_boxes]
        repo_simple_climb = RepositorySimpleClimb.get_instance()
        repo_datasets = RepositoryDatasets.get_instance()
        dataset_id = self.get_dataset_id(bbox_list)
        if not ignore_cache:
            try:
                data = repo_datasets.get_dataset(self.builder_name, dataset_id)
                n_train = len(data["input_maps_train"])
                n_test = len(data["input_maps_test"])
                logger.info("Loaded dataset from cache (%s train, %s test patches)", n_train, n_test)
                return data
            except ValueError:
                pass

        all_climbs: list[SimpleClimb] = []
        seen_ids: set[str] = set()
        for bbox in bbox_list:
            climbs = await repo_simple_climb.get_all_in_bounding_box_by_ground(bbox, verbose=True)
            for c in climbs:
                cid = c.simple_climb_id
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_climbs.append(c)
        climbs = all_climbs
        if len(climbs) < 2:
            raise ValueError(f"Need at least 2 SimpleClimbs across regions, got {len(climbs)}")

        if self.dataset_builder_type == "flat_seed":
            train_result, test_result, n_zero_climb = self._build_dataset_flat_seed_multi(bbox_list, climbs)
        elif len(bbox_list) == 1:
            train_climbs, test_climbs = self.split_climbs(climbs)
            train_result, test_result, n_zero_climb = self.build(bbox_list[0], train_climbs, test_climbs)
        else:
            train_climbs, test_climbs = self.split_climbs(climbs)
            train_result, test_result, n_zero_climb = self._build_from_multiple_bboxes(
                bbox_list, train_climbs, test_climbs
            )
        train_result, test_result = self._apply_dataset_limit(train_result, test_result)
        data: dict = {
            "input_maps_train": train_result.input_maps,
            "target_maps_train": train_result.target_maps,
            "input_maps_test": test_result.input_maps,
            "target_maps_test": test_result.target_maps,
            "lats_train": train_result.lats,
            "lons_train": train_result.lons,
            "lats_test": test_result.lats,
            "lons_test": test_result.lons,
            "metadata": {
                "strength_lo": train_result.strength_lo,
                "strength_hi": train_result.strength_hi,
                "image_size": self.image_size,
                "patch_size_m": self.patch_size_m,
                "grid_stride": self.grid_stride,
                "dataset_mode": self.dataset_builder_type,
                "bounding_boxes": [b.model_dump() for b in bbox_list],
            },
        }
        data["dataset_id"] = dataset_id
        data["time_of_day_train"] = train_result.time_of_day
        data["time_of_year_train"] = train_result.time_of_year
        data["time_of_day_test"] = test_result.time_of_day
        data["time_of_year_test"] = test_result.time_of_year
        data["ground_alt_train"] = train_result.ground_alt
        data["start_alt_train"] = train_result.start_alt
        data["end_alt_train"] = train_result.end_alt
        data["ground_alt_test"] = test_result.ground_alt
        data["start_alt_test"] = test_result.start_alt
        data["end_alt_test"] = test_result.end_alt

        repo_datasets.save_dataset(self.builder_name, dataset_id, data)
        logger.info(
            "Cached dataset (%s train, %s test patches, %s zero-climb points) to repository",
            len(train_result.input_maps),
            len(test_result.input_maps),
            n_zero_climb,
        )
        return data
