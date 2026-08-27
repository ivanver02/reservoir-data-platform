FROM python:3.12-slim

WORKDIR /app

# The model bundled with Prophet needs the OpenMP runtime when it fits
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend backend

# Data is mounted here, with the source files under /data/raw
ENV RESERVOIR_DATA_ROOT=/data
# Scripts mounted under /data still import the backend package from /app
ENV PYTHONPATH=/app

# Arguments after the service name become CLI commands
ENTRYPOINT ["python", "-m", "backend.cli"]
