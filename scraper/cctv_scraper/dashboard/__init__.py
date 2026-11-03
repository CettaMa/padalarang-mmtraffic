"""Read-only monitoring dashboard for the CCTV scraper.

The dashboard never writes into the dataset tree.  It reads the status CSVs,
raw recorded segments, daily 7z archives, logs, and disk usage that the recorder
already produces, and keeps its own SQLite roll-up so uptime history survives
log rotation and ``RETENTION_DAYS`` pruning.
"""
