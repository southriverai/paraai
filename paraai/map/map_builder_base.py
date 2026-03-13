"""Map builder: estimate climb maps from points using convolution."""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from paraai.map.vectror_map_array import VectorMapArray

logger = logging.getLogger(__name__)


@dataclass
class MapEvaluateResult:
    """Result of evaluate."""

    count_mae: float
    count_rmse: float
    strength_mae: float
    strength_rmse: float
    n_train: int
    n_holdout: int

    # def __str__(self) -> str:
    #     result = f"Count MAE: {self.count_mae:.4f}\n"
    #     result += f"Count RMSE: {self.count_rmse:.4f}\n"
    #     result += f"Strength MAE: {self.strength_mae:.4f}\n"
    #     result += f"Strength RMSE: {self.strength_rmse:.4f}\n"
    #     result += f"N train: {self.n_train}\n"
    #     result += f"N holdout: {self.n_holdout}\n"
    #     result += f"N flat removed: {self.n_flat_removed}\n"
    #     return result


class MapBuilderBase:
    def __init__(
        self,
        name: str,
        output_map_names: list[str] | None = None,
    ):
        self.name = name
        self.output_map_names = output_map_names if output_map_names is not None else ["strength", "count"]

    def get_cache_params(self) -> dict:
        """Params for cache key. Override in subclasses (e.g. kernel_size_m)."""
        return {}

    def build(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        df: pd.DataFrame,
        *,
        ignore_cache: bool = False,
    ) -> dict[str, VectorMapArray]:
        """Build or load from cache. When ignore_cache=False, returns cached maps if available."""
        if not ignore_cache:
            maps = self._try_load_from_cache(lat_min, lat_max, lon_min, lon_max)
            if maps is not None:
                logger.info("Loaded %s maps from cache", self.name)
                return maps
        maps = self._build_impl(lat_min, lat_max, lon_min, lon_max, df)
        self.save_maps(maps, lat_min, lat_max, lon_min, lon_max, **self.get_cache_params())
        return maps

    def _try_load_from_cache(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> dict[str, VectorMapArray] | None:
        """Try to load all maps from cache. Returns None if any missing."""
        from paraai.repository.repository_maps import RepositoryMaps

        repo = RepositoryMaps.get_instance()
        params = self.get_cache_params()
        maps: dict[str, VectorMapArray] = {}
        for name in self.output_map_names:
            vma = repo.get_map(
                self.name,
                name,
                lat_min,
                lat_max,
                lon_min,
                lon_max,
                **params,
            )
            if vma is None:
                return None
            maps[name] = vma
        return maps

    @abstractmethod
    def _build_impl(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        df: pd.DataFrame,
    ) -> dict[str, VectorMapArray]:
        """Build maps. Override in subclasses."""
        pass

    def save_map(
        self,
        vma: VectorMapArray,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        *,
        file_path: str | Path | None = None,
        use_cache: bool = True,
        **builder_params: object,
    ) -> Path:
        """Save map to file or cache. If use_cache, saves via RepositoryMaps. Returns path."""
        if use_cache:
            from paraai.repository.repository_maps import RepositoryMaps

            repo = RepositoryMaps.get_instance()
            path = repo.save_map(
                vma,
                generator_name=self.name,
                map_name=vma.map_name,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                **builder_params,
            )
            logger.info("Cached map %s/%s to %s", self.name, vma.map_name, path)
            return path
        if file_path is None:
            raise ValueError("file_path required when use_cache=False")
        import rasterio

        with rasterio.open(str(file_path), "w", **vma.profile) as dst:
            dst.write(vma.array, 1)
        return Path(file_path)

    def save_maps(
        self,
        maps: dict[str, VectorMapArray],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        *,
        use_cache: bool = True,
        **builder_params: object,
    ) -> dict[str, Path]:
        """Save multiple maps to cache. Returns dict of map_name -> path."""
        return {
            name: self.save_map(vma, lat_min, lat_max, lon_min, lon_max, use_cache=use_cache, **builder_params)
            for name, vma in maps.items()
        }

    def evaluate(
        self,
        count_map: VectorMapArray,
        strength_map: VectorMapArray,
        evaluate_df: pd.DataFrame,
        n_train: int,
    ) -> MapEvaluateResult:
        """Evaluate estimated climb maps on held-out points. DataFrame must have lat and lon columns."""
        if evaluate_df.empty:
            return MapEvaluateResult(
                count_mae=0.0,
                count_rmse=0.0,
                strength_mae=0.0,
                strength_rmse=0.0,
                n_train=n_train,
                n_holdout=0,
            )
        if "lat" not in evaluate_df.columns or "lon" not in evaluate_df.columns:
            raise ValueError("evaluate_df must have columns 'lat' and 'lon'")
        count_col = "count" if "count" in evaluate_df.columns else None
        strength_col = "strength" if "strength" in evaluate_df.columns else None

        count_errors: list[float] = []
        strength_errors: list[float] = []
        for _, row in evaluate_df.iterrows():
            lat, lon = row["lat"], row["lon"]
            count = row[count_col] if count_col else 1.0
            strength = row[strength_col] if strength_col else 0.0
            pred_count = count_map.sample(lat, lon)
            pred_strength = strength_map.sample(lat, lon)
            count_errors.append(abs(pred_count - count))
            strength_errors.append(abs(pred_strength - strength))

        return MapEvaluateResult(
            count_mae=float(np.mean(count_errors)),
            count_rmse=float(np.sqrt(np.mean([e**2 for e in count_errors]))),
            strength_mae=float(np.mean(strength_errors)),
            strength_rmse=float(np.sqrt(np.mean([e**2 for e in strength_errors]))),
            n_train=n_train,
            n_holdout=len(evaluate_df),
        )
