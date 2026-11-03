# CCTV 24/7 Scraper

This project records public HLS CCTV streams continuously with FFmpeg and collects TomTom traffic
and Open-Meteo weather metadata alongside them. Completed dataset days are packaged as-is into
verified, uncompressed 7-Zip Store archives for transfer to a personal computer -- there is no
local re-encoding step. The dataset is kept on local disk for thesis analysis; there is no Google
Drive upload component.

## Requirements

- Python 3.10 or newer
- FFmpeg on `PATH` for local runs
- 7-Zip (`7z`) on `PATH` for daily archiving
- Docker and Docker Compose for the containerized setup

Install the Python package and development tools locally with:

```bash
python -m pip install -e ".[dev]"
```

## Configure camera points

Create `cctv_points.csv` in the project directory. A header is recommended:

```csv
name,url,lat,lon
Camera One,https://example.invalid/live.m3u8,-6.850,107.500
Camera Two,https://example.invalid/other.m3u8,-6.851,107.501
```

`name` becomes the sanitized dataset directory name. `url` must be an HTTP or HTTPS URL. If
`lat` or `lon` is omitted, the configured default coordinates are used. Camera URLs are treated
as credentials by the operator: rotating or expired `memfs` URLs must be replaced in this file.

## Configure `.env`

Copy `.env.example` to `.env` and set the stream, archive, storage, and metadata options you need.
The most important settings are:

```env
CCTV_CONFIG_FILE=cctv_points.csv
CCTV_OUTPUT_ROOT=dataset
FFMPEG_TRANSPORT_MODE=copy
ARCHIVE_SAFE_AGE_SECONDS=90
DAILY_ARCHIVE_ENABLED=true
DAILY_ARCHIVE_SCAN_SECONDS=300
DAILY_ARCHIVE_DELETE_SOURCE=false
ARCHIVER_BINARY=7z
TOMTOM_API=your_tomtom_key
```

`DAILY_ARCHIVE_DELETE_SOURCE=false` (the default) retains both the raw `.ts` segments and the
7z archive, which is useful while validating a deployment; set it to `true` once you trust the
archive to replace the source. `RETENTION_DAYS`, `MIN_FREE_SPACE_GB`, `LOG_MAX_BYTES`, and
`LOG_BACKUP_COUNT` control storage protection. See `.env.example` for all supported variables.

## Run locally

With `.env` and `cctv_points.csv` present:

```bash
python main.py
```

The CLI flags remain available and override their corresponding environment values:

```bash
python main.py --config cctv_points.csv --output dataset \
  --segment-seconds 60 --retention-days 7 --min-free-space-gb 20 --video-container ts
```

Stop with `Ctrl-C`. FFmpeg recording, metadata collection, daily archiving, disk monitoring,
and status reporting run as separate workers.

## Run with Docker

The Compose service mounts the local dataset and camera configuration into the container:

```bash
docker compose up -d --build
docker compose logs -f
```

Use `--build` whenever the source has changed. Both services share the
`cctv-scraper:latest` tag, and a plain `docker compose up -d` reuses whatever image already
carries that tag instead of rebuilding it, which silently runs yesterday's code.

There is no local hardware/software encoding step and therefore no GPU device to pass through --
the image runs identically on any host. The recorder needs no encoder at all in the default
`copy` transport mode; only `FFMPEG_TRANSPORT_MODE=smooth` or `transcode` invoke one, and that
runs on the CPU via `VIDEO_ENCODER` (e.g. `libx264`).

Stop the services with:

```bash
docker compose down
```

## Monitoring dashboard

A read-only FastAPI dashboard reports recorder health from the files the recorder already
writes: status CSVs, raw recorded segments, daily 7z archives, logs, and disk usage. It runs as
its own process so a dashboard failure can never interrupt a recording that cannot be repeated.

Install the extra dependencies and run it locally:

```bash
python -m pip install -e ".[dashboard]"
python dashboard.py
```

Then open `http://localhost:8080`.

Configure it through the `DASHBOARD_*` variables in `.env` (see `.env.example`). Camera `memfs`
URLs are treated as credentials, so the dashboard refuses to start when
`DASHBOARD_AUTH_ENABLED=true` and `DASHBOARD_PASSWORD` is empty. To run without a password, set
`DASHBOARD_AUTH_ENABLED=false` **and** `DASHBOARD_HOST=127.0.0.1`. Every URL leaving the process
is masked (`https://host/memfs/****c09e5.m3u8`), including the URLs quoted back inside FFmpeg and
`requests` error strings.

### What it reports

- **Uptime (24 h)** reconstructed from status transitions. `online_preflight_ok` is written once
  per FFmpeg start, so a healthy camera is silent for hours; liveness therefore comes from raw
  segment mtimes, and uptime stops accruing once segments stop arriving.
- **Coverage** per camera per day, computed directly from raw `.ts` segment filenames (each
  segment is up to `SEGMENT_SECONDS` of footage). This is the recording-completeness figure
  worth citing in analysis.
- **Backlog**: `.ts` segments recorded but not yet folded into a completed daily 7z.
- **Storage**: free space against `MIN_FREE_SPACE_GB`, plus the daily `.7z` archives and any
  that failed to package.
- **Incidents**: non-online status rows across all cameras, including
  `EXPIRED_URL_ESCALATION`.

### History

The dashboard keeps a SQLite roll-up (`DASHBOARD_HISTORY_DB`, sampled every
`DASHBOARD_HISTORY_INTERVAL_SECONDS`) so uptime and coverage survive `RETENTION_DAYS` pruning and
log rotation. Past dates are walked once and then finalized. This database is the only thing the
dashboard writes; it lives outside the date-based directories so `DiskMonitor` never prunes it.

### Run with Docker

The `cctv-dashboard` Compose service reuses the recorder image with a different command and
mounts the dataset **read-only**, so it cannot damage a recording it is reporting on. Its SQLite
state lives in the `dashboard-data` volume.

```bash
docker compose up -d --build cctv-dashboard
```

Publish a different host port with `DASHBOARD_HOST_PORT` if 8080 is taken.

### API

`GET /healthz` is unauthenticated for container health checks. Everything under `/api/` requires
Basic Auth when it is enabled: `/api/overview`, `/api/cameras/{name}`, `/api/coverage?days=N`,
`/api/history/uptime`, `/api/history/disk`, and `/api/logs`.

## Dataset layout

```text
dataset/
├── YYYY-MM-DD/
│   └── <camera-name>/
│       ├── videos/
│       │   └── <camera>_YYYYMMDD_HHMMSS.ts
│       └── metadata/
│           └── <camera>_YYYY-MM-DD_metadata.csv
├── archives/
│   ├── YYYY-MM-DD.7z
│   └── .YYYY-MM-DD.7z.failure.json   (only present if packaging failed)
├── logs/
│   ├── scraper.log
│   ├── <camera>.log
│   └── ffmpeg/<camera>_YYYY-MM-DD.ffmpeg.log
├── status/
│   └── <camera>_status.csv
└── ...
```

Raw `.ts` segments are written continuously and never re-encoded locally. After a date changes,
the daily archiver packages the complete previous date as-is into `archives/YYYY-MM-DD.7z` using
7-Zip Store (`-mx=0`), verifies it with `7z t`, and retries it on the next scan if needed; a
persistently failing day leaves a hidden `.YYYY-MM-DD.7z.failure.json` marker beside the archive
and keeps its raw segments untouched. Source directories are retained by default; set
`DAILY_ARCHIVE_DELETE_SOURCE=true` only if the verified archive should replace the source.
Logs and status files are retained according to `RETENTION_DAYS`;
application and per-camera logs also rotate by size.

Metadata CSVs retain the traffic and weather cache-status columns (`fresh`, `cached`, and
`fresh_fallback`) used by downstream analysis. The recorder status CSV records online/offline
transitions and emits `EXPIRED_URL_ESCALATION` when an expired URL remains unavailable for the
configured number of consecutive preflight checks.
