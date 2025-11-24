# main.py
import os
import json
import uuid
import numpy as np
from shapely.geometry import shape, Point, mapping
import rasterio
from rasterio.mask import mask
from rasterio.transform import xy as raster_xy
from rasterio.warp import reproject, Resampling
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO, BytesIO
import csv
import geopandas as gpd
import requests

# ------------------------
# CONFIG
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEM_PATH = os.path.join(BASE_DIR, "data", "srtm_cgiar_nepal_boundary.img")
LULC_PATH = os.path.join(BASE_DIR, "data", "lc2022.tif")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

GRID_SIZE = 80  # Sampling resolution

# ------------------------
# GOOGLE DRIVE DEM DOWNLOAD
# ------------------------
def download_from_gdrive(file_id, dest_path):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# DEM
if not os.path.exists(DEM_PATH):
    print("DEM not found, downloading from Google Drive...")
    file_id = "113sRzSWz9PQUBrysiCu6Mo_wUqI6Kp3G"  # Your link
    download_from_gdrive(file_id, DEM_PATH)
    print("DEM downloaded successfully.")

# LULC
if not os.path.exists(LULC_PATH):
    raise FileNotFoundError(f"LULC not found at {LULC_PATH}")

# ------------------------
# APP INIT
# ------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/data", StaticFiles(directory=os.path.join(BASE_DIR, "data")), name="data")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ------------------------
# HELPER FUNCTIONS
# ------------------------
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

def compute_flood_prob(elev, slope, lc, rainfall):
    """Compute flood probability"""
    slope_norm = float(slope)/90.0
    lc_scores = {1:1.0,2:0.2,3:0.15,4:0.2,5:0.95,6:0.9,7:0.7,8:0.8,9:0.85,10:0.45,11:0.35}
    lc_score = lc_scores.get(int(lc), 0.5)
    max_rain = 200.0
    rain_norm = min(1.0,max(0.0,rainfall/max_rain))
    flood_prob = 0.45*rain_norm + 0.35*slope_norm + 0.2*lc_score
    return max(0.0,min(1.0,flood_prob))

def process_aoi(poly, rainfall=80.0, grid_size=GRID_SIZE):
    dem_crop, dem_transform = mask(dem_raster,[mapping(poly)],crop=True)
    lulc_crop, _ = mask(lulc_raster,[mapping(poly)],crop=True)
    if dem_crop.size==0:
        raise ValueError("AOI does not intersect DEM/LULC")
    dem_arr = dem_crop[0]
    lulc_arr = lulc_crop[0]
    slope_crop = compute_slope_from_dem(dem_arr)
    rows, cols = dem_arr.shape

    flood_grid = np.full((rows,cols),np.nan)
    for i in range(rows):
        for j in range(cols):
            elev = dem_arr[i,j]
            if np.isnan(elev):
                continue
            lc_val = int(lulc_arr[i,j]) if not np.isnan(lulc_arr[i,j]) else -1
            slope_val = float(slope_crop[i,j])
            f_val = compute_flood_prob(elev, slope_val, lc_val, rainfall)
            flood_grid[i,j]=f_val

    flood_grid = adjust_flood_prob_with_neighbors(np.nan_to_num(flood_grid,nan=0.0), alpha=0.6)
    row_indices = np.linspace(0, rows-1, min(grid_size,rows)).astype(int)
    col_indices = np.linspace(0, cols-1, min(grid_size,cols)).astype(int)

    features=[]
    for ri in row_indices:
        for cj in col_indices:
            lon, lat = raster_xy(dem_transform,ri,cj,offset="center")
            pt = Point(lon, lat)
            if not poly.contains(pt): continue
            f_val = float(flood_grid[ri,cj]) if not np.isnan(flood_grid[ri,cj]) else 0.0
            features.append({
                "type":"Feature",
                "geometry":mapping(pt),
                "properties":{
                    "flood_prob":f_val
                }
            })
    summary = {
        "points_sampled": len(features),
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
# LOAD DEM & LULC
# ------------------------
dem_raster = rasterio.open(DEM_PATH)
lulc_raster = rasterio.open(LULC_PATH)
lulc_resampled = np.zeros_like(dem_raster.read(1),dtype=lulc_raster.read(1).dtype)
reproject(
    source=lulc_raster.read(1),
    destination=lulc_resampled,
    src_transform=lulc_raster.transform, src_crs=lulc_raster.crs,
    dst_transform=dem_raster.transform, dst_crs=dem_raster.crs,
    resampling=Resampling.nearest
)

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
    except:
        raise HTTPException(400,"Invalid AOI")
    rainfall = float(payload.get("rainfall",80.0))
    fc,aoi_feature,summary = process_aoi(poly,rainfall=rainfall)
    return JSONResponse({"feature_collection":fc,"aoi":aoi_feature,"summary":summary})

@app.post("/export_csv")
async def export_csv(request: Request):
    payload = await request.json()
    if "aoi" not in payload:
        raise HTTPException(400,"Missing 'aoi'")
    try:
        poly = shape(payload["aoi"]) if not ("type" in payload["aoi"] and payload["aoi"]["type"]=="Feature") else shape(payload["aoi"]["geometry"])
    except:
        raise HTTPException(400,"Invalid AOI")
    rainfall = float(payload.get("rainfall",80.0))
    fc,aoi_feature,summary = process_aoi(poly,rainfall=rainfall)
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["latitude","longitude","flood_prob"])
    for feat in fc.get("features", []):
        lon, lat = feat["geometry"]["coordinates"]
        f_prob = feat["properties"]["flood_prob"]
        writer.writerow([lat, lon, f_prob])
    csv_bytes = output.getvalue().encode("utf-8")
    fname = f"export_{uuid.uuid4().hex}.csv"
    return StreamingResponse(BytesIO(csv_bytes), media_type="text/csv", headers={"Content-Disposition":f"attachment; filename={fname}"})

@app.post("/upload_geojson")
async def upload_geojson(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".geojson"):
        raise HTTPException(400,"Please upload a geojson file")
    data = json.loads((await file.read()).decode("utf-8"))
    gdf = gpd.GeoDataFrame.from_features(data["features"])
    if gdf.crs is None:
        gdf.set_crs(epsg=4326,inplace=True)
    elif gdf.crs.to_epsg()!=4326:
        gdf = gdf.to_crs(4326)
    geom = gdf.unary_union
    minx, miny, maxx, maxy = geom.bounds
    features=[]
    num_points=500
    for _ in range(num_points):
        lat = np.random.uniform(miny,maxy)
        lon = np.random.uniform(minx,maxx)
        pt = Point(lon,lat)
        if not geom.contains(pt):
            continue
        features.append({
            "type":"Feature",
            "geometry":{"type":"Point","coordinates":[lon,lat]},
            "properties":{
                "flood_prob":float(np.random.random())
            }
        })
    summary={
        "points_sampled":len(features),
        "mean_flood_prob":float(np.mean([f["properties"]["flood_prob"] for f in features])) if features else 0
    }
    return JSONResponse({"aoi": data,"features":{"type":"FeatureCollection","features":features},"summary":summary})

# ------------------------
# RUN
# ------------------------
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
