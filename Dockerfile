FROM python:3.12-slim

WORKDIR /app

# Install system dependencies.
# The pango/cairo libraries are WeasyPrint's text and raster backends, used for
# the daily PDF snapshots. WeasyPrint rather than a headless browser because
# these pages carry no JavaScript, so Chromium would be ~400MB to lay out
# static HTML.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Heavy ML wheels first, in their own layer: they are ~450MB and almost never
# change, so edits to the light requirements below reuse this layer.
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# Everything else
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/

# Set Python path
ENV PYTHONPATH=/app/src

# Create data and snapshot directories
RUN mkdir -p /app/src/mlb_hr/data /app/snapshots

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run app
# Fetch any data files missing from the volume (no-op without HF keys).
CMD ["sh", "-c", "python -m mlb_hr.hf_data pull; exec python -m mlb_hr.app"]
