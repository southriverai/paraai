import asyncio
from pathlib import Path

from tqdm import tqdm

from paraai.repository.repository_tracklog_body import RepositoryTracklogBody
from paraai.tools_plot import plot_tracklog_modifications
from paraai.tools_tracklogbody import clean_tracklog


async def clean_tracklogbody_by_id(tracklog_id: str) -> None:
    repository_tracklog_body = RepositoryTracklogBody.get_instance()
    tracklog_body = repository_tracklog_body.get_raise(tracklog_id)
    clean_tracklog_body, clean_tracklog_results = clean_tracklog(tracklog_body)
    plot_tracklog_modifications(tracklog_body, clean_tracklog_body, clean_tracklog_results)

    print(clean_tracklog_results)


async def clean_all():
    print("Cleaning all tracklogs")
    repository_tracklog_body = RepositoryTracklogBody.get_instance()
    tracklog_bodies = await repository_tracklog_body.asample(10)
    for tracklog_body in tqdm(tracklog_bodies, desc="Cleaning tracklogs", total=len(tracklog_bodies)):
        print(tracklog_body.tracklog_id)
        await clean_tracklogbody_by_id(tracklog_body.tracklog_id)
    print("Cleaning complete")


if __name__ == "__main__":
    path_dir_database = Path("data", "database_sqlite")
    repository_tracklog_body = RepositoryTracklogBody.initialize_sqlite(path_dir_database)
    asyncio.run(clean_all())
