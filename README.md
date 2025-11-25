# Flood and Heat Index Simulator for Nepal AOIs

A **FastAPI**-based web service that simulates **flood probability** and **heat index** for any Area of Interest (AOI) in Nepal. The app uses **DEM (Digital Elevation Model)** and **Land Use/Land Cover (LULC)** data to generate spatially sampled points with calculated flood and heat risk.  

---

## Features

- Accepts AOI in **GeoJSON** format and calculates:
  - Heat Index (temperature + rainfall adjusted by slope)
  - Flood Probability (based on rainfall, slope, and land cover)
- Provides:
  - **GeoJSON FeatureCollection** of sampled points
  - **CSV export** of results
- Automatically downloads required **DEM and LULC datasets** from Google Drive if missing.
- Adjustable **rainfall** and **temperature** inputs.
- Designed for **interactive mapping or GIS integration**.

---

## DEM Data

 **Don't forget to add the DEM data (.img) to the `data/` folder.**  
   Download it from [DEM Data](https://www.mediafire.com/file/5amrotgjt6twxzn/srtm_cgiar_nepal_boundary.img/file)

---

## Demo

Example endpoints:  

- **Root**: `/` → Serves `index.html` (basic interface)  
- **Simulate AOI**: `/simulate` (POST JSON AOI + optional rainfall/temperature)  
- **Export CSV**: `/export_csv` (POST JSON AOI + optional rainfall/temperature)  
- **Upload GeoJSON**: `/upload_geojson` (POST `.geojson` file for random sample simulation)

---

## Installation (Local)

1. Clone the repository:

```bash
git clone https://github.com/yourusername/flood-heat-nepal.git
cd flood-heat-nepal
```

2. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the app:

```bash
uvicorn main:app --reload
```

---

## License
This project is under the MIT License.
