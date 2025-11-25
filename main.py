# main.py
import os
import json
import uuid
import numpy as np
from shapely.geometry import shape, Point, mapping
import rasterio
from rasterio.mask import mask
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO, BytesIO
import csv
import geopandas as gpd
import requests
import subprocess

# ------------------------
# CONFIG
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

DEM_IMG_TEMP = os.path.join(DATA_DIR, "srtm_temp.img")
DEM_PATH = os.path.join(DATA_DIR, "srtm_cgiar_nepal_boundary.tif")
LULC_PATH = os.path.join(DATA_DIR, "lc2022.tif")
GRID_SIZE = 80

# MediaFire DEM link (replace with yours)
DEM_MEDIAFIRE_URL = "https://www.mediafire.com/file/5amrotgjt6twxzn/srtm_cgiar_nepal_boundary.img/file"

# ------------------------
# FASTAPI SETUP
# ------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ------------------------
# HELPER FUNCTIONS
# ------------------------
def download_from_mediafire(url, dest_path):
    """Download file from MediaFire by following redirects."""
    session = requests.Session()
    resp = session.get(url, stream=True, allow_redirects=True, timeout=60)
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(32768):
            if chunk:
                f.write(chunk)
    size_mb = os.path.getsize(dest_path)/(1024*1024)
    if size_mb < 1:  # DEM should be larger than 1 MB
        raise RuntimeError("Downloaded file too small, likely HTML or invalid download.")
    print(f"Downloaded file from MediaFire: {dest_path} (~{size_mb:.2f} MB)")

def convert_img_to_tif(img_path, tif_path):
    """Convert .img raster to GeoTIFF using GDAL."""
    try:
        subprocess.run(["gdal_translate", "-of", "GTiff", img_path, tif_path], check=True)
        print(f"Converted {img_path} to GeoTIFF: {tif_path}")
    except Exception as e:
        raise RuntimeError(f"GDAL conversion failed: {e}")

def verify_raster(path):
    """Check if Rasterio can open the file"""
    try:
        with rasterio.open(path) as ds:
            print(f"Raster {path} opened successfully. Size: {ds.width}x{ds.height}")
    except Exception as e:
        raise RuntimeError(f"Failed to open raster {path}: {e}")

def compute_slope_from_dem(dem_array, xres=30.0, yres=30.0):
    dz_dy, dz_dx = np.gradient(dem_array.astype(float), yres, xres)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    return np.degrees(slope_rad)

def adjust_flood_prob_with_neighbors(flood_grid, alpha=0.6):
    nx, ny = flood_grid.shape
    adj_grid = flood_grid.copy()
    for i in range(nx):
        for j in range(ny):
            neighbors = []
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    if dx==0 and dy==0: continue
                    ni, nj = i+dx, j+dy
                    if 0<=ni<nx and 0<=nj<ny:
                        neighbors.append(float(flood_grid[ni,nj]))
            if neighbors:
                neighbor_avg = sum(neighbors)/len(neighbors)
                adj_grid[i,j] = alpha*float(flood_grid[i,j]) + (1-alpha)*neighbor_avg
    return adj_grid

def compute_rain_heat(elev, slope, lc, rainfall, temperature):
    slope_norm = float(slope)/90.0
    lc_scores = {1:1.0,2:0.2,3:0.15,4:0.2,5:0.95,6:0.9,7:0.7,8:0.8,9:0.85,10:0.45,11:0.35}
    lc_score = lc_scores.get(int(lc), 0.5)
    max_rain = 200.0
    rain_norm = min(1.0,max(0.0,rainfall/max_rain))
    hi = temperature + 0.02*rainfall - 0.01*slope_norm*100
    flood_prob = 0.45*rain_norm + 0.35*slope_norm + 0.2*lc_score
    flood_prob = max(0.0,min(1.0,flood_prob))
    return round(hi,3), round(flood_prob,4)

def process_aoi(poly, rainfall=80.0, temperature=28.0, grid_size=GRID_SIZE):
    """Process AOI and generate heat_index & flood_prob grid"""
    with rasterio.open(DEM_PATH) as dem_raster:
        dem_crop, dem_transform = mask(dem_raster, [mapping(poly)], crop=True)
    with rasterio.open(LULC_PATH) as lulc_raster:
        lulc_crop, _ = mask(lulc_raster, [mapping(poly)], crop=True)
    if dem_crop.size==0:
        raise ValueError("AOI does not intersect DEM/LULC")
    
    dem_arr = dem_crop[0]
    lulc_arr = lulc_crop[0]
    slope_crop = compute_slope_from_dem(dem_arr)
    rows, cols = dem_arr.shape
    hi_grid = np.full((rows,cols),np.nan)
    flood_grid = np.full((rows,cols),np.nan)

    for i in range(rows):
        for j in range(cols):
            elev = dem_arr[i,j]
            if np.isnan(elev):
                continue
            try:
                lc_val = int(lulc_arr[i,j]) if not np.isnan(lulc_arr[i,j]) else -1
            except:
                lc_val=-1
            slope_val = float(slope_crop[i,j])
            hi, fprob = compute_rain_heat(elev,slope_val,lc_val,rainfall,temperature)
            hi_grid[i,j] = hi
            flood_grid[i,j] = fprob

    flood_grid = adjust_flood_prob_with_neighbors(np.nan_to_num(flood_grid,nan=0.0), alpha=0.6)

    row_indices = np.linspace(0, rows-1, min(grid_size, rows)).astype(int)
    col_indices = np.linspace(0, cols-1, min(grid_size, cols)).astype(int)

    features=[]
    for ri in row_indices:
        for cj in col_indices:
            lon, lat = rasterio.transform.xy(dem_transform, ri, cj, offset="center")
            pt = Point(lon, lat)
            if not poly.contains(pt): continue
            hi_val = float(hi_grid[ri,cj]) if not np.isnan(hi_grid[ri,cj]) else 0.0
            f_val = float(flood_grid[ri,cj]) if not np.isnan(flood_grid[ri,cj]) else 0.0
            f_val = max(0.0,min(1.0,f_val))
            features.append({
                "type":"Feature",
                "geometry":mapping(pt),
                "properties":{
                    "heat_index":hi_val,
                    "flood_prob":f_val
                }
            })
    summary = {
        "points_sampled": len(features),
        "mean_heat_index": float(np.mean([f["properties"]["heat_index"] for f in features])) if features else 0,
        "mean_flood_prob": float(np.mean([f["properties"]["flood_prob"] for f in features])) if features else 0
    }
    aoi_feature = {
        "type":"Feature",
        "geometry":mapping(poly),
        "properties":summary
    }
    fc = {"type":"FeatureCollection","features":features}
    return fc, aoi_feature, summary

# ------------------------
# PREPARE DEM
# ------------------------
if not os.path.exists(DEM_PATH):
    print("DEM not found. Downloading from MediaFire...")
    download_from_mediafire(DEM_MEDIAFIRE_URL, DEM_IMG_TEMP)
    convert_img_to_tif(DEM_IMG_TEMP, DEM_PATH)
    os.remove(DEM_IMG_TEMP)
verify_raster(DEM_PATH)

# ------------------------
# ROUTES
# ------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(STATIC_DIR,"index.html")
    if os.path.exists(index_path):
        with open(index_path,"r",encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>index.html NOT FOUND</h1>")

@app.post("/simulate")
async def simulate(request: Request):
    payload = await request.json()
    if "aoi" not in payload:
        raise HTTPException(400,"Missing 'aoi'")
    try:
        poly = shape(payload["aoi"]) if not ("type" in payload["aoi"] and payload["aoi"]["type"]=="Feature") else shape(payload["aoi"]["geometry"])
    except Exception as e:
        raise HTTPException(400,f"Invalid AOI or Outside the boundary")
    rainfall = float(payload.get("rainfall",80.0))
    temperature = float(payload.get("temperature",28.0))
    fc,aoi_feature,summary = process_aoi(poly,rainfall=rainfall,temperature=temperature)
    return JSONResponse({"feature_collection":fc,"aoi":aoi_feature,"summary":summary})

# ------------------------
# RUN
# ------------------------
if __name__=="__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
