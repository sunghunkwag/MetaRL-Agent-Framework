# ---- Build Stage ----
FROM python:3.11 AS builder

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY core /app/core
COPY meta_rl /app/meta_rl
COPY env_runner /app/env_runner
COPY adaptation /app/adaptation

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build && \
    python -m build --wheel --outdir dist .

# ---- Final Stage ----
FROM python:3.11-slim

WORKDIR /app

# Install system deps required by gymnasium/mujoco
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/dist/*.whl /tmp/

RUN pip install --no-cache-dir /tmp/*.whl && \
    rm /tmp/*.whl

# Copy application scripts (main, experiments, etc.)
COPY main.py /app/main.py
COPY experiments /app/experiments

# No ENTRYPOINT, allow user to specify script via docker run command
# Example: docker run <image_name> python main.py --env_name Pendulum-v1
# Example: docker run <image_name> python experiments/quick_benchmark.py

# Default command (optional, can be overridden)
CMD ["python", "main.py", "--help"]
