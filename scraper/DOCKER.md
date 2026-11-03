# Docker Usage

## Build and run

```powershell
docker compose up -d --build
```

Always include `--build` when the source has changed. Both services share the
`cctv-scraper:latest` tag, so a plain `docker compose up -d` reuses whatever image already
carries that tag instead of rebuilding it -- silently running yesterday's code.

Logs:

```powershell
docker compose logs -f
```

Stop:

```powershell
docker compose down
```

## FFmpeg and encoding

The image does not contain NVIDIA CUDA, Intel VA-API/QuickSync drivers, or require any GPU
passthrough. There is no local re-encoding step: raw `.ts` segments are packaged as-is into the
daily 7z archive. The recorder itself needs no encoder in the default `copy` transport mode.

Check the available FFmpeg encoders (only relevant if you set `FFMPEG_TRANSPORT_MODE=smooth` or
`transcode`, which run on the CPU via `VIDEO_ENCODER`, e.g. `libx264`):

```powershell
docker compose run --rm cctv-scraper ffmpeg -hide_banner -encoders
```

## Data and config

The image does not bake `.env` or `dataset` into the container.

Runtime mounts:

```text
./dataset -> /app/dataset
./cctv_points.csv -> /app/cctv_points.csv
```

Runtime env:

```text
.env -> container environment (local recording configuration only)
TZ=Asia/Jakarta -> container timezone
```

## Local video storage

Google Drive uploading has been removed. All recordings remain in the local dataset directory:

```text
dataset/<date>/<camera>/videos/*.ts
dataset/<date>/<camera>/metadata/*.csv
dataset/archives/<date>.7z
```

The Compose configuration mounts `./dataset` to `/app/dataset`, so files remain available on the
host. The image also includes `7z`; completed dates are written as verified Store archives in
`dataset/archives/YYYY-MM-DD.7z`. Configure `DAILY_ARCHIVE_SCAN_SECONDS` to change the scan
interval. Source date directories are retained by default; set `DAILY_ARCHIVE_DELETE_SOURCE=true`
only after confirming the transfer workflow.
