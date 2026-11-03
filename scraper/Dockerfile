# ---------------------------------------------------------------------------
# Stage 1: Build dependencies inside a virtual environment
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt requirements-dashboard.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dashboard.txt

# ---------------------------------------------------------------------------
# Stage 2: Minimal runtime image with FFmpeg for HLS capture and 7z for archiving
# ---------------------------------------------------------------------------
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Jakarta \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        7zip \
        gosu \
        python3 \
        tzdata \
        tini \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

COPY main.py dashboard.py ./
COPY cctv_scraper/ ./cctv_scraper/
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY cctv_points.csv ./

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /app/dataset /app/dashboard-data \
    && chown -R appuser:appuser /app

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["python3", "main.py"]