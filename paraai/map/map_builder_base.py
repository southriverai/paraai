"""Map builder: estimate climb maps from points using convolution."""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox
from paraai.tool_string import validate_safe_name

logger = logging.getLogger(__name__)


@dataclass
class MapEvaluateResult:
    """Result of evaluate."""

    strength_mae: float
    strength_rmse: float
    n_holdout: int
    true_values: np.ndarray | None = None
    pred_values: np.ndarray | None = None

    def __str__(self) -> str:
        result = f"Strength MAE: {self.strength_mae:.4f}\n"
        result += f"Strength RMSE: {self.strength_rmse:.4f}\n"
        result += f"N holdout: {self.n_holdout}\n"
        return result


class MapBuilderBase:
    def __init__(
        self,
        name: str,
        output_map_names: list[str],
    ):
        validate_safe_name(name)
        self.name = name
        if len(output_map_names) == 0:
            raise ValueError("output_map_names must be a non-empty list")
        self.output_map_names = output_map_names

    @abstractmethod
    def get_builder_id(self) -> str:
        pass

    def build(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame | None = None,
        *,
        ignore_cache: bool = False,
        model_id: str | None = None,
    ) -> dict[str, VectorMapArray]:
        """Build or load from cache. When ignore_cache=False, returns cached maps if available."""
        if not ignore_cache:
            maps = self._try_load_from_cache(bounding_box)
            if maps is not None:
                logger.info("Loaded %s maps from cache", self.name)
                return maps
        maps = self._build_impl(bounding_box, df, model_id=model_id)
        self.save_maps(maps, bounding_box)
        return maps

    def _try_load_from_cache(self, bounding_box: BoundingBox) -> dict[str, VectorMapArray] | None:
        """Try to load all maps from cache. Returns None if any missing."""
        from paraai.repository.repository_maps import RepositoryMaps

        repo = RepositoryMaps.get_instance()
        maps: dict[str, VectorMapArray] = {}
        for name in self.output_map_names:
            vma = repo.get_map(
                self.name,
                name,
                bounding_box,
                builder_id=self.get_builder_id(),
            )
            if vma is None:
                return None
            maps[name] = vma
        return maps

    @abstractmethod
    def _build_impl(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame | None = None,
        model_id: str | None = None,
    ) -> dict[str, VectorMapArray]:
        """Build maps. Override in subclasses."""
        pass

    def save_map(
        self,
        vma: VectorMapArray,
        bounding_box: BoundingBox,
    ) -> Path:
        """Save map to cache via RepositoryMaps. Returns path."""
        from paraai.repository.repository_maps import RepositoryMaps

        repo = RepositoryMaps.get_instance()
        path = repo.save_map(
            vma,
            generator_name=self.name,
            map_name=vma.map_name,
            bounding_box=bounding_box,
            builder_id=self.get_builder_id(),
        )
        logger.info("Cached map %s/%s to %s", self.name, vma.map_name, path)
        return path

    def save_maps(
        self,
        maps: dict[str, VectorMapArray],
        bounding_box: BoundingBox,
    ) -> dict[str, Path]:
        """Save multiple maps to cache. Returns dict of map_name -> path."""
        return {name: self.save_map(vma, bounding_box) for name, vma in maps.items()}

    def evaluate(
        self,
        vector_map: VectorMapArray,
        evaluate_df: pd.DataFrame,
        column_name: str,
    ) -> MapEvaluateResult:
        """Evaluate estimated climb maps on held-out points. DataFrame must have lat and lon columns."""
        if len(evaluate_df) == 0:
            raise ValueError("evaluate_df is empty")
        if "lat" not in evaluate_df.columns or "lon" not in evaluate_df.columns or column_name not in evaluate_df.columns:
            raise ValueError("evaluate_df must have columns 'lat', 'lon', and 'column_name'")
        lats = evaluate_df["lat"].values
        lons = evaluate_df["lon"].values
        true_values = evaluate_df[column_name].values
        pred_values = vector_map.get_values(lats, lons)
        errors = abs(pred_values - true_values)

        return MapEvaluateResult(
            strength_mae=float(np.mean(errors)),
            strength_rmse=float(np.sqrt(np.mean([e**2 for e in errors]))),
            n_holdout=len(evaluate_df),
            true_values=true_values,
            pred_values=pred_values,
        )
