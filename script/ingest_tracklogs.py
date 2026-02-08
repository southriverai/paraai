from pathlib import Path

from tqdm import tqdm

from paraai.model.tracklog import parse_igc_bytes
from paraai.repository.repository_tracklog_body import RepositoryTracklogBody
from paraai.repository.repository_tracklog_header import RepositoryTracklogHeader, TracklogHeaderQueryRequest

if __name__ == "__main__":
    # configuration
    limit_tracklogs = -1  # 1000
    path_dir_tracklog_igc = Path("C:\\project\\glide\\glide-data\\syride_track_log_igc")
    path_dir_database = Path("data", "database_sqlite")

    repository_tracklog_body = RepositoryTracklogBody.initialize_sqlite(path_dir_database)
    repository_tracklog_header = RepositoryTracklogHeader.initialize_sqlite(path_dir_database)

    path_files_igc: list[Path] = list(path_dir_tracklog_igc.glob("*.igc"))
    path_files_igc = path_files_igc[:limit_tracklogs]
    for path_file_igc in tqdm(path_files_igc, desc="Ingesting tracklogs"):
        filename = path_file_igc.name
        # check if tracklog already exists
        tracklog_header = repository_tracklog_header.query(TracklogHeaderQueryRequest(filename=filename))
        if tracklog_header:
            continue
        tracklog_header, tracklog_body = parse_igc_bytes(path_file_igc.name, path_file_igc.read_bytes())
        repository_tracklog_body.insert(tracklog_body)
        repository_tracklog_header.insert(tracklog_header)
