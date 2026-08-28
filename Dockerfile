# Alqueva 24h Energy Trading Optimizer — container image.
#
# No CPLEX license is baked into this image (it's commercial, per-machine
# licensed software — see config/solver.yaml). The pipeline still runs: the
# solver fallback chain (config/solver.yaml: fallback_order) automatically
# uses the free HiGHS solver (via the `highspy` package, installed below)
# whenever CPLEX isn't found. To use CPLEX inside this container, install
# and license it separately and mount/set config/solver.yaml's `executable`
# path accordingly — nothing else changes.
#
# Default CMD runs the test suite as a build-time-equivalent smoke test:
# `docker run <image>` proves the image installs cleanly and the
# non-solver-dependent code passes, exactly like CI (.github/workflows/tests.yml)
# but from a container instead of a runner. Override CMD to run the actual
# pipeline, e.g.:
#   docker run <image> python run_production.py --date auto --auto --synthetic

FROM python:3.12-slim

# libgomp1: OpenMP runtime required by xgboost/lightgbm at import time on
# Debian slim images (not needed on the full python image, needed here).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "pytest", "tests/", "-v"]
