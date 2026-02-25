"""Ingest IGC tracklog files into the database. Resumable (skips existing) and supports parallel parsing."""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

from tqdm import tqdm

from paraai.model.tracklog import TracklogBody, TracklogHeader, parse_igc_bytes
from paraai.repository.repository_tracklog_body import RepositoryTracklogBody
from paraai.repository.repository_tracklog_header import (
    RepositoryTracklogHeader,
    TracklogHeaderQueryRequest,
)


def _parse_igc_file(path_str: str) -> tuple[str, TracklogHeader, TracklogBody] | None:
    """Parse one IGC file. Returns (filename, header, body) or None on failure. Top-level for pickling."""
    try:
        path = Path(path_str)
        header, body = parse_igc_bytes(path.name, path.read_bytes())
        return (path.name, header, body)
    except Exception:
        return None


def _worker_parse(in_queue: mp.Queue, out_queue: mp.Queue) -> None:
    """Worker: read paths from in_queue, parse, put results in out_queue. Exits on None."""
    while True:
        path_str = in_queue.get()
        if path_str is None:
            break
        result = _parse_igc_file(path_str)
        out_queue.put((path_str, result))


def main() -> None:
    # Force unbuffered output for progress visibility when run via poetry/subprocess
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Ingest IGC tracklog files into the database")
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        default=Path("C:\\project\\glide\\glide-data\\syride_track_log_igc"),
        help="Directory containing .igc files",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=-1,
        help="Max number of files to process (-1 = all)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-ingest existing files (delete before insert). Default: skip existing (resumable)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel workers for parsing (1 = sequential)",
    )
    parser.add_argument(
        "--database",
        "-d",
        type=Path,
        default=Path("data", "database_sqlite"),
        help="Database directory",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="Write progress to this file (e.g. for tail -f when terminal output is buffered)",
    )
    args = parser.parse_args()

    path_dir_tracklog_igc = args.path
    path_dir_database = args.database
    limit_tracklogs = args.limit if args.limit > 0 else None
    force = args.force
    n_workers = max(1, args.workers)
    progress_file = args.progress_file

    def log(msg: str, clear_file: bool = False) -> None:
        """Print to stderr and optionally write to progress file."""
        print(msg, file=sys.stderr, flush=True)
        if progress_file:
            try:
                mode = "w" if clear_file else "a"
                with progress_file.open(mode) as f:
                    f.write(msg + "\n")
                    f.flush()
            except OSError:
                pass

    log("Ingest tracklogs starting...", clear_file=True)
    repository_tracklog_body = RepositoryTracklogBody.initialize_sqlite(path_dir_database)
    repository_tracklog_header = RepositoryTracklogHeader.initialize_sqlite(path_dir_database)

    path_files_igc: list[Path] = sorted(path_dir_tracklog_igc.glob("*.igc"))
    if limit_tracklogs:
        path_files_igc = path_files_igc[:limit_tracklogs]
    log(f"Found {len(path_files_igc)} .igc files")

    # Resumability: skip files already ingested (unless --force)
    BATCH_SIZE = 500
    if not force:
        log("Checking which files are already ingested...")
        to_process: list[Path] = []
        batches = [
            path_files_igc[i : i + BATCH_SIZE]
            for i in range(0, len(path_files_igc), BATCH_SIZE)
        ]
        for batch in tqdm(batches, desc="Checking", unit="batch", file=sys.stderr):
            filenames = [p.name for p in batch]
            response = repository_tracklog_header.query(
                TracklogHeaderQueryRequest(filenames=filenames, limit=len(filenames))
            )
            existing = {t.file_name for t in response.tracklogs}
            for path_file in batch:
                if path_file.name not in existing:
                    to_process.append(path_file)
        skipped = len(path_files_igc) - len(to_process)
        if skipped:
            log(f"Skipping {skipped} files already ingested (use --force to re-ingest)")
        path_files_igc = to_process
    else:
        log("Force mode: will delete and re-insert existing files")

    if not path_files_igc:
        log("No files to process")
        return

    total = len(path_files_igc)
    log(f"Processing {total} files with {n_workers} worker(s)...")
    path_strings = [str(p) for p in path_files_igc]

    report_interval = min(100, max(1, total // 200))  # Report every 100 files or 0.5%, whichever is smaller

    if n_workers == 1:
        # Sequential: main process does everything
        for i, path_str in enumerate(path_strings):
            _ingest_one(path_str, repository_tracklog_header, repository_tracklog_body, force)
            if (i + 1) % report_interval == 0 or (i + 1) == total:
                log(f"  {i + 1}/{total} files")
    else:
        # Parallel: child processes parse, main process inserts and prints (all output from main)
        in_queue: mp.Queue = mp.Queue()
        out_queue: mp.Queue = mp.Queue()
        for path_str in path_strings:
            in_queue.put(path_str)
        for _ in range(n_workers):
            in_queue.put(None)  # Sentinel per worker

        workers = [
            mp.Process(target=_worker_parse, args=(in_queue, out_queue))
            for _ in range(n_workers)
        ]
        log(f"Starting {n_workers} worker processes...")
        for w in workers:
            w.start()
        log("Worker processes started, processing files...")

        done = 0
        pbar = tqdm(total=total, desc="Ingesting", unit="file", file=sys.stderr)
        while done < total:
            _, result = out_queue.get()
            if result is not None:
                filename, tracklog_header, tracklog_body = result
                _insert_one(
                    filename,
                    tracklog_header,
                    tracklog_body,
                    repository_tracklog_header,
                    repository_tracklog_body,
                    force,
                )
            done += 1
            pbar.update(1)
            if done % report_interval == 0 or done == total:
                log(f"  {done}/{total} files")

        pbar.close()
        for w in workers:
            w.join()

    log("Done.")


def _ingest_one(
    path_str: str,
    repository_tracklog_header: RepositoryTracklogHeader,
    repository_tracklog_body: RepositoryTracklogBody,
    force: bool,
) -> None:
    """Parse and insert one file (sequential path)."""
    result = _parse_igc_file(path_str)
    if result is None:
        return
    filename, tracklog_header, tracklog_body = result
    _insert_one(
        filename,
        tracklog_header,
        tracklog_body,
        repository_tracklog_header,
        repository_tracklog_body,
        force,
    )


def _insert_one(
    filename: str,
    tracklog_header: TracklogHeader,
    tracklog_body: TracklogBody,
    repository_tracklog_header: RepositoryTracklogHeader,
    repository_tracklog_body: RepositoryTracklogBody,
    force: bool,
) -> None:
    """Delete existing (if force) and insert. Called from main process only."""
    if force:
        response = repository_tracklog_header.query(TracklogHeaderQueryRequest(filename=filename))
        for existing in response.tracklogs:
            tracklog_id = existing.tracklog_id
            repository_tracklog_body.store.delete(tracklog_id)
            repository_tracklog_body.store_clean.delete(tracklog_id)
            repository_tracklog_header.store.delete(tracklog_id)
    repository_tracklog_body.insert(tracklog_body)
    repository_tracklog_header.insert(tracklog_header)


if __name__ == "__main__":
    main()
