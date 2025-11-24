# --------------------------
# Base Image
# --------------------------
FROM python:3.10-slim

# --------------------------
# Install system dependencies for GDAL, Rasterio, Shapely
# --------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    proj-bin \
    libgeos-dev \
    python3-dev \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal

# --------------------------
# Set Work Directory
# --------------------------
WORKDIR /app

# --------------------------
# Copy dependency list
# --------------------------
COPY requirements.txt .

# --------------------------
# Install Python dependencies
# --------------------------
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# --------------------------
# Copy application
# --------------------------
COPY . .

# Ensure data/static dirs exist
RUN mkdir -p /app/data /app/static

# --------------------------
# Expose port (Railway/Render will map automatically)
# --------------------------
EXPOSE 8000

# --------------------------
# Start FastAPI server
# --------------------------
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
