# Comprehensive Approach: Day-Level Correlation & Distance Interaction

**Goal:** Test whether (a) climbs are correlated per day within a region ("strong day" vs "weak day"), and (b) whether there is an interaction with distance between consecutive thermals.

**Constraint:** Millions of climbs → prefer streaming/iterative processing; GPU where applicable.

---

## 1. Hypotheses

| Hypothesis | Question | If true |
|------------|----------|---------|
| **H1: Day-level correlation** | Do climbs in a region on the same day share strength? | Need hierarchical model: sample day quality first, then climb strength |
| **H2: Distance interaction** | Is spacing between thermals correlated with strength? (e.g. strong days → thermals closer together, or vice versa) | Need joint model of (strength, spacing) conditional on day |
| **H3: Region-specific** | Does day correlation vary by region (valley, coastal, etc.)? | Stratify analysis by region |

---

## 2. Data Requirements

From `SimpleClimb`:
- `tracklog_id`, `start_timestamp_utc`, `start_lat`, `start_lon`, `start_alt_m`, `end_alt_m`, `end_timestamp_utc`
- Vertical speed: `(end_alt_m - start_alt_m) / (end_timestamp_utc - start_timestamp_utc)`
- **Distance between thermals**: only meaningful within a tracklog (same flight). Use haversine between consecutive climb centers, or time gap as proxy.

---

## 3. Architecture: Streaming + DuckDB

**Why DuckDB:**
- Single-pass aggregation over millions of rows
- Native Parquet support (streaming read)
- Window functions for distance-between (LAG over ordered climbs)
- No full load into memory
- Can export results to Pandas/numpy for GPU stats if needed

**Pipeline:**
```
[SQLite DB] → (stream export) → [Parquet] → [DuckDB SQL] → [aggregated tables] → [Python stats/viz]
```

Alternative: stream directly from repo in batches, write to Parquet in chunks, then DuckDB. Avoids loading all keys at once.

---

## 4. Phase 1: Export to Parquet (streaming)

Stream climbs from repo in batches (e.g. 10k–50k per batch). For each climb emit:
```
tracklog_id, start_timestamp_utc, date_str, month, doy,
lat, lon, region_bin, vertical_speed_m_s
```
- `region_bin`: e.g. `(round(lat*10), round(lon*10))` → ~3600 Europe bins, or coarser
- `date_str`: `YYYY-MM-DD` for grouping

**Memory:** O(batch_size) only. Write to Parquet in row groups (PyArrow/DuckDB).

**Optional:** If repo supports ordered iteration (e.g. by tracklog_id), we could stream in tracklog order and avoid a separate sort for Phase 3.

---

## 5. Phase 2: Day-Level Variance Decomposition (ICC)

**Input:** Parquet with (date, region, speed).

**Single DuckDB query:**
```sql
WITH daily AS (
  SELECT date_str, region_bin, month,
         AVG(vertical_speed_m_s) AS day_mean,
         VAR_POP(vertical_speed_m_s) AS day_var,
         COUNT(*) AS n
  FROM climbs
  WHERE vertical_speed_m_s > 0 AND vertical_speed_m_s < 10
  GROUP BY date_str, region_bin, month
  HAVING COUNT(*) >= 3
)
SELECT
  region_bin,
  month,
  AVG(day_var) AS within_var,
  VAR_POP(day_mean) AS between_var,
  (VAR_POP(day_mean)) / (AVG(day_var) + VAR_POP(day_mean)) AS icc
FROM daily
GROUP BY region_bin, month;
```

**ICC (intraclass correlation):** `between_var / (within_var + between_var)`. High ICC → strong day effect.

**Streaming alternative:** If Parquet is too large to scan at once, use Polars lazy + streaming:
```python
pl.scan_parquet("climbs.parquet").group_by(["date_str", "region_bin"]).agg(...)
```

---

## 6. Phase 3: Distance-Between-Thermals

**Input:** Climbs per tracklog, ordered by time.

**Approach:** DuckDB window functions.
```sql
WITH ordered AS (
  SELECT *,
         LAG(start_lat) OVER w AS prev_lat,
         LAG(start_lon) OVER w AS prev_lon,
         LAG(start_timestamp_utc) OVER w AS prev_ts
  FROM climbs
  WINDOW w AS (PARTITION BY tracklog_id ORDER BY start_timestamp_utc)
),
with_dist AS (
  SELECT *,
         haversine_km(lat, lon, prev_lat, prev_lon) AS dist_km,
         (start_timestamp_utc - prev_ts) / 60.0 AS gap_min
  FROM ordered WHERE prev_lat IS NOT NULL
)
SELECT date_str, region_bin, tracklog_id,
       AVG(dist_km) AS mean_dist_km,
       AVG(vertical_speed_m_s) AS mean_speed,
       AVG(gap_min) AS mean_gap_min
FROM with_dist
GROUP BY date_str, region_bin, tracklog_id
HAVING COUNT(*) >= 2;
```

Need a UDF or inline haversine. DuckDB supports `acos(sin(lat1)*sin(lat2)+cos(lat1)*cos(lat2)*cos(lon2-lon1))*6371`.

**Output:** Per-tracklog-per-day: (mean_distance_between, mean_speed, date, region).

**Interaction test:** 
- Correlation(day_mean_speed, day_mean_distance) across days?
- Or: within a day, do tracklogs with shorter spacing have stronger thermals?

---

## 7. Phase 4: Analyses and Visualizations

| Analysis | Method | Output |
|---------|--------|--------|
| ICC by region | Phase 2 query | Map or bar chart: ICC per region |
| ICC by month | Phase 2 query | Seasonal variation in day effect |
| Day mean vs day std | Scatter | Do strong days have different spread? |
| Distribution of day means | Histogram | Bimodal → strong/weak days |
| Autocorrelation of day means | Lag-1, Lag-2 correlation | Multi-day streaks |
| Speed vs distance (within tracklog) | Scatter / regression | Interaction |
| Speed vs distance (day-level) | Correlation by region | Regional interaction |

---

## 8. GPU / Iterative Considerations

| Stage | Approach | Rationale |
|-------|----------|-----------|
| Export | Batched Python loop | Repo has batch API; no GPU benefit |
| Parquet write | PyArrow (batched) | Columnar, compressed |
| Aggregation | DuckDB on Parquet | Single pass, C++ optimised |
| ICC, correlations | NumPy/SciPy on aggregated | Aggregated data small (~50k rows) |
| Optional GPU | CuPy/cuDF for correlation matrix | If 1000s of regions × lags |

**Streaming keys:** Avoid `list(repo.store.yield_keys())` with millions of keys. Use:
```python
def iter_keys_batched(repo, batch_size=5000):
    batch = []
    for k in repo.store.yield_keys():  # generator
        batch.append(k)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
```
Then `mget` each batch. Memory stays O(batch_size).

---

## 9. Implementation Plan

1. **`export_climbs_to_parquet.py`** – Stream repo → Parquet. Columns: tracklog_id, ts, date, month, doy, lat, lon, region_bin, speed. CLI: `--limit`, `--region europe`.
2. **`analyze_day_correlation.py`** – Run Phase 2 + 3 queries, save ICC and per-tracklog stats to CSV.
3. **`plot_day_correlation.py`** – Load results, produce all Phase 4 plots.

Or combine 2+3 into one script with `--analyse` / `--plot` flags.

---

## 10. Open Questions

1. **Region definition:** Grid (0.1°?) vs clustering (leader-follower like show_day)? Grid is simpler for streaming.
2. **UTC vs local:** Thermals follow sun. Consider converting to local solar time by lon?
3. **Minimum day size:** Need ≥3 climbs per (date, region) for variance. Tune threshold.
4. **Distance metric:** Haversine is proxy. True distance along glide path would need full tracklog—out of scope for SimpleClimb.

---

## 11. Dependencies

- **DuckDB** (`duckdb`) – fast single-pass aggregation; add via `poetry add duckdb`
- **PyArrow** (`pyarrow`) – Parquet write/read; `poetry add pyarrow`
- NumPy, SciPy, matplotlib – already in project

Alternatively: **Polars** (`polars`) has streaming and can replace DuckDB for most queries, but DuckDB SQL is simpler for window functions (LAG). Could use both: Polars for export, DuckDB for analysis.

---

## 12. Expected Outcomes

- **If ICC high (e.g. > 0.2):** Implement hierarchical sampler in tools_sim.
- **If distance correlation significant:** Add spacing to thermal model (strong days ↔ different spacing).
- **If regional variation:** Consider region-specific parameters.
