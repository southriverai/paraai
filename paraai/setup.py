import logging
from pathlib import Path

from paraai.repository.repository_cache import RepositoryCache
from paraai.repository.repository_datasets import RepositoryDatasets
from paraai.repository.repository_maps import RepositoryMaps
from paraai.repository.repository_models import RepositoryModels
from paraai.repository.repository_simple_climb import RepositorySimpleClimb
from paraai.repository.repository_simple_climb_pixel import RepositorySimpleClimbPixel
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.repository.repository_tracklog_body import RepositoryTracklogBody
from paraai.repository.repository_tracklog_header import RepositoryTracklogHeader

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("srai_store").setLevel(logging.WARNING)
    logger.info("Logging setup complete")


def setup_database():
    logger.info("Setting up database")
    path_sqlite = Path("data", "database_sqlite")
    path_cache = Path("data", "database_cache")
    path_terrain = Path("data", "database_terrain")
    path_maps = Path("data", "database_maps")
    path_datasets = Path("data", "database_datasets")
    path_models = Path("data", "database_models")
    RepositorySimpleClimb.initialize_sqlite(path_sqlite)
    RepositorySimpleClimbPixel.initialize_sqlite(path_sqlite)
    RepositoryTracklogBody.initialize_sqlite(path_sqlite)
    RepositoryTracklogHeader.initialize_sqlite(path_sqlite)
    RepositoryCache.initialize(path_cache)
    RepositoryTerrain.initialize(path_terrain)
    RepositoryMaps.initialize(path_maps)
    RepositoryDatasets.initialize(path_datasets)
    RepositoryModels.initialize(path_models)
    logger.info("Database setup complete")


def setup():
    setup_logging()
    setup_database()
