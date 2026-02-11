"""Show takeoff lat/lon for a given date. Example: May 18, 2025."""

import asyncio
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paraai.repository.repository_tracklog_header import RepositoryTracklogHeader

TARGET_DATE = "2025-05-18"
TARGET_SITE_LAT = 45.30933
TARGET_SITE_LNG = 5.89027
TARGET_SITE_RADIUS_KM = 50
CLUSTER_THRESHOLD_KM = 0.5  # Takeoffs within 500 m are same site


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Distance in km between two lat/lng points."""
    R = 6371  # Earth radius km
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def leader_follower_cluster(lats: list[float], lngs: list[float], threshold_km: float) -> list[int]:
    """Assign cluster labels via Leader-Follower: points within threshold of a leader form same cluster."""
    if not lats:
        return []
    labels = [0] * len(lats)
    centers_lat = [lats[0]]
    centers_lng = [lngs[0]]
    for i in range(1, len(lats)):
        lat, lng = lats[i], lngs[i]
        min_d = float("inf")
        best_j = -1
        for j in range(len(centers_lat)):
            d = _haversine_km(lat, lng, centers_lat[j], centers_lng[j])
            if d < min_d:
                min_d = d
                best_j = j
        if min_d <= threshold_km:
            labels[i] = best_j
            n = sum(1 for k in range(i + 1) if labels[k] == best_j)
            # Update center as running mean
            centers_lat[best_j] = (centers_lat[best_j] * (n - 1) + lat) / n
            centers_lng[best_j] = (centers_lng[best_j] * (n - 1) + lng) / n
        else:
            labels[i] = len(centers_lat)
            centers_lat.append(lat)
            centers_lng.append(lng)
    return labels


async def amain():
    path_dir_database = Path("data", "database_sqlite")
    repo = RepositoryTracklogHeader.initialize_sqlite(path_dir_database)
    headers = await repo.get_all()

    on_date = [h for h in headers if h.date == TARGET_DATE]
    on_date = [h for h in on_date if _haversine_km(h.takeoff_lat, h.takeoff_lng, TARGET_SITE_LAT, TARGET_SITE_LNG) <= TARGET_SITE_RADIUS_KM]
    on_date.sort(key=lambda h: (h.takeoff_lat, h.takeoff_lng))

    if not on_date:
        print(f"No takeoffs within {TARGET_SITE_RADIUS_KM} km of ({TARGET_SITE_LAT}, {TARGET_SITE_LNG}) on {TARGET_DATE}")
        return

    lats = [h.takeoff_lat for h in on_date]
    lngs = [h.takeoff_lng for h in on_date]
    labels = leader_follower_cluster(lats, lngs, CLUSTER_THRESHOLD_KM)
    n_clusters = max(labels) + 1

    # Order sites by unique pilots (desc), then takeoff count (desc)
    def site_stats(c: int) -> tuple[int, int, int]:
        subset = [h for h, lb in zip(on_date, labels) if lb == c]
        n_takeoffs = len(subset)
        n_pilots = len({h.pilot_name for h in subset})
        total_seconds = sum(h.duration_seconds for h in subset)
        return (-n_pilots, -n_takeoffs, total_seconds)

    order = sorted(range(n_clusters), key=lambda c: (site_stats(c)[0], site_stats(c)[1]))

    def _format_duration(seconds: int) -> str:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h{m:02d}m" if h else f"{m}m{s}s"

    print(
        f"Takeoffs on {TARGET_DATE} (within {TARGET_SITE_RADIUS_KM} km of site): {len(on_date)}, {n_clusters} sites (threshold {CLUSTER_THRESHOLD_KM} km)"
    )
    for rank, cluster_id in enumerate(order, start=1):
        subset = [h for h, lb in zip(on_date, labels) if lb == cluster_id]
        lat_c = np.mean([h.takeoff_lat for h in subset])
        lng_c = np.mean([h.takeoff_lng for h in subset])
        n_pilots = len({h.pilot_name for h in subset})
        total_seconds = sum(h.duration_seconds for h in subset)
        print(
            f"  #{rank} Site {cluster_id + 1}: ({lat_c:.5f}, {lng_c:.5f}) pilots={n_pilots} takeoffs={len(subset)} flight_time={_format_duration(total_seconds)}"
        )
        for h in subset:
            print(f"    {h.takeoff_lat:.6f}, {h.takeoff_lng:.6f}  ({h.pilot_name})")

    cmap = plt.colormaps["tab10"]
    fig, ax = plt.subplots(figsize=(10, 8))
    for rank, cluster_id in enumerate(order, start=1):
        mask = np.array(labels) == cluster_id
        ax.scatter(
            np.array(lngs)[mask],
            np.array(lats)[mask],
            color=cmap((cluster_id % 10) / 10),
            alpha=0.7,
            s=50,
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Takeoff locations on {TARGET_DATE} (within {TARGET_SITE_RADIUS_KM} km, {n_clusters} sites)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    asyncio.run(amain())
