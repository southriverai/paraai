import asyncio
from pathlib import Path

from tqdm import tqdm

from paraai.repository.repository_climb import RepositoryClimb
from paraai.repository.repository_tracklog_body import RepositoryTracklogBody
from paraai.repository.repository_tracklog_header import RepositoryTracklogHeader
from paraai.tools_tracklogbody import clean_tracklog, extract_climbs


def _safe_takeoff_utc(header) -> int | None:
    try:
        return header.takeoff_timestamp_utc
    except Exception:
        return None


async def main():
    path_dir_database = Path("data", "database_sqlite")
    repo_header = RepositoryTracklogHeader.initialize_sqlite(path_dir_database)
    repo_body = RepositoryTracklogBody.get_instance()
    repo_climb = RepositoryClimb.initialize_sqlite(path_dir_database)

    headers = await repo_header.get_all()
    total_climbs = 0

    for header in tqdm(headers, desc="Collecting climbs"):
        takeoff_utc = _safe_takeoff_utc(header)
        if takeoff_utc is None:
            continue

        try:
            tracklog_body = repo_body.get_raise(header.tracklog_id)
        except Exception:
            continue

        clean_body, _ = clean_tracklog(tracklog_body)
        climbs = extract_climbs(clean_body, takeoff_timestamp_utc=takeoff_utc)

        repo_climb.delete_climbs_for_tracklog(header.tracklog_id)
        repo_climb.insert_many(climbs)
        total_climbs += len(climbs)

    print(f"Collected {total_climbs} climbs from {len(headers)} tracklogs")


if __name__ == "__main__":
    path_dir_database = Path("data", "database_sqlite")
    RepositoryTracklogBody.initialize_sqlite(path_dir_database)
    asyncio.run(main())
