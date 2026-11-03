"""Uvicorn launcher for the monitoring dashboard.

The dashboard runs as its own process (and its own Compose service) so that a
dashboard failure can never take the 24/7 recorder down with it.
"""

import logging
import sys

from cctv_scraper.dashboard.settings import DashboardConfigError, load_dashboard_config


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("dashboard")

    try:
        config = load_dashboard_config()
    except DashboardConfigError as exc:
        logger.error("%s", exc)
        return 2

    import uvicorn

    logger.info(
        "Serving dashboard on http://%s:%s (auth=%s, history=%s).",
        config.host,
        config.port,
        "on" if config.auth_enabled else "off",
        "on" if config.history_enabled else "off",
    )
    uvicorn.run(
        "cctv_scraper.dashboard.api:create_app",
        factory=True,
        host=config.host,
        port=config.port,
        log_level="info",
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
