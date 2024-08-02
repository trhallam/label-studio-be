# syntax=docker/dockerfile:1

FROM rapidsai/miniforge-cuda:cuda11.8.0-base-ubuntu22.04-py3.10

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8

# Update the base OS
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt install --no-install-recommends -y  \
    wget git; \
    apt-get autoremove -y

# setup environment
COPY condarc /opt/conda/.condarc
COPY conda-env.yml .

RUN --mount=type=cache,target="/opt/conda/pkgs",sharing=locked \
    /opt/conda/bin/conda env create -f /app/conda-env.yml

# copy program
COPY _wsgi.py .
# COPY model.py .
# COPY sam_predictor.py .

EXPOSE 9090

CMD conda run -n label-studio-be \
    --no-capture-output \
    gunicorn \
    --bind :${PORT:-9090} \
    --workers $WORKERS \
    --threads $THREADS \
    --timeout 0 \
    _wsgi:app
