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
import shutil

# ------------------------
# CONFIG
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

DEM_PATH = os.path.join(DATA_DIR, "srtm_cgiar_nepal_boundary.img")
LULC_PATH = os.path.join(DATA_DIR, "lc2022.tif")
GRID_SIZE = 80  # sampling resolution

# Google Drive file IDs (replace with your actual file IDs)
DEM_GDRIVE_ID = "113sRzSWz9PQUBrysiCu6Mo_wUqI6Kp3G"
LULC_GDRIVE_ID = "YOUR_LULC_FILE_ID"

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
def download_from_gdrive(file_id, dest_path):
    """Download a large file from Google Drive with confirmation."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(url, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    total = int(response.headers.get('Content-Length', 0))
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    print(f"Downloaded {dest_path} successfully.")

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
    # Read only clipped DEM & LULC
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
# DOWNLOAD FILES IF MISSING
# ------------------------
if not os.path.exists(DEM_PATH):
    print("DEM not found, downloading from Google Drive...")
    download_from_gdrive(DEM_GDRIVE_ID, DEM_PATH)
if not os.path.exists(LULC_PATH):
    print("LULC not found, downloading from Google Drive...")
    download_from_gdrive(LULC_GDRIVE_ID, LULC_PATH)

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

@app.post("/export_csv")
async def export_csv(request: Request):
    payload = await request.json()
    if "aoi" not in payload:
        raise HTTPException(400,"Missing 'aoi'")
    try:
        poly = shape(payload["aoi"]) if not ("type" in payload["aoi"] and payload["aoi"]["type"]=="Feature") else shape(payload["aoi"]["geometry"])
    except Exception as e:
        raise HTTPException(400,f"Invalid AOI GeoJSON: {e}")
    rainfall = float(payload.get("rainfall",80.0))
    temperature = float(payload.get("temperature",28.0))
    fc,aoi_feature,summary = process_aoi(poly,rainfall=rainfall,temperature=temperature)
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["latitude","longitude","heat_index","flood_prob"])
    for feat in fc.get("features", []):
        lon, lat = feat["geometry"]["coordinates"]
        props = feat["properties"]
        writer.writerow([lat, lon, props["heat_index"], props["flood_prob"]])
    
    csv_bytes = output.getvalue().encode("utf-8")
    fname = f"export_{uuid.uuid4().hex}.csv"
    return StreamingResponse(BytesIO(csv_bytes),
                             media_type="text/csv",
                             headers={"Content-Disposition":f"attachment; filename={fname}"})

@app.post("/upload_geojson")
async def upload_geojson(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".geojson", ".json")):
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
                "heat_index":float(np.random.uniform(20,45)),
                "flood_prob":float(np.random.random())
            }
        })
    summary={
        "points_sampled":len(features),
        "mean_heat_index":float(np.mean([f["properties"]["heat_index"] for f in features])) if features else 0,
        "mean_flood_prob":float(np.mean([f["properties"]["flood_prob"] for f in features])) if features else 0
    }
    return JSONResponse({
        "aoi": data,
        "features":{"type":"FeatureCollection","features":features},
        "summary":summary
    })

# ------------------------
# RUN
# ------------------------
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
