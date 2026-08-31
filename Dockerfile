# Alqueva 24h Energy Trading Optimizer — container image.
#
# No CPLEX license is baked into this image (it's commercial, per-machine
# licensed software — see config/solver.yaml). This project's solving is
# CPLEX-only by design; config/solver.yaml lists a HiGHS/CBC fallback_order,
# but it is not what's actually verified/relied on for correctness here. To
# actually solve inside this container, install and license CPLEX separately
# and mount/set config/solver.yaml's `executable` path accordingly.
#
# Default CMD runs the test suite as a build-time-equivalent smoke test:
# `docker run <image>` proves the image installs cleanly and the
# non-solver-dependent code passes (every test that actually solves the MILP
# skips gracefully without a CPLEX license — see tests/test_reserve_*.py's
# `cfg.solver.resolve_executable() is None` guards), exactly like CI
# (.github/workflows/tests.yml) but from a container instead of a runner.
# Override CMD to run the actual pipeline (needs a licensed CPLEX to solve):
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
