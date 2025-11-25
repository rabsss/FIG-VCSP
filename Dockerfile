# Dockerfile
# --------------------------
# Base Image
# --------------------------
FROM python:3.10-slim

# --------------------------
# Install system dependencies for GDAL, Rasterio, Shapely, GeoPandas
# --------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    proj-bin \
    libgeos-dev \
    libspatialindex-dev \
    python3-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# GDAL include paths for compiling wheels
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal

WORKDIR /app

# Copy dependency list
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application
COPY . /app

# Ensure data/static dirs exist
RUN mkdir -p /app/data /app/static

EXPOSE 8000

# Start FastAPI server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
