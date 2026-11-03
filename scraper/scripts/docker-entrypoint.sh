#!/usr/bin/env bash
set -Eeuo pipefail

# Dataset is a bind mount, so image-time ownership does not apply to it.
# Fix the mounted directory before dropping privileges to appuser.
mkdir -p /app/dataset 2>/dev/null || true
if [[ -w /app/dataset ]]; then
    chown appuser:appuser /app/dataset
    # Date directories created outside this container -- by a host-side run, or by
    # an image whose appuser had a different UID -- stay root-owned and silently
    # block every write once we drop privileges. Touch only the entries that are
    # not already appuser's, so an already-correct dataset costs one scan and no
    # writes rather than a full recursive chown on every start.
    find /app/dataset ! -user appuser -exec chown appuser:appuser {} + 2>/dev/null || true
else
    # The dashboard service mounts the dataset read-only; there is nothing to fix.
    echo "Dataset is mounted read-only; skipping ownership fix." >&2
fi

# Own the dashboard's private state volume when this container has one.
if [[ -d /app/dashboard-data && -w /app/dashboard-data ]]; then
    chown appuser:appuser /app/dashboard-data
fi

exec gosu appuser "$@"
