# 🌍 SAMPLE: High-Resolution Climate Digital Twin & AI Forecasting Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM%20%7C%20XGBoost-orange.svg)](https://lightgbm.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Dataset-IMD%2075--Year%20Gridded%20Data-green.svg)](https://imdpune.gov.in/)
[![Web Dashboard](https://img.shields.io/badge/Dashboard-RIT%20INDIA-cyan.svg)](http://localhost:8765)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**RIT** is an end-to-end Machine Learning Framework and Interactive Web Application for high-resolution daily meteorological predictions across India ($0.25^\circ \times 0.25^\circ$ IMD grid).

The system integrates 75 years of historical observational data, oceanic climate drivers (NOAA ONI & IOD DMI), topographic DEM elevations, ancient Indian Astro-Meteorological indicators (*Bhadali Subhashitas*), real-world satellite observations (INSAT-3D LST), and an active web interface.

---

## 🌟 Key Capabilities & Model Architecture

```
                                  ┌─────────────────────────────────────────┐
                                  │   IMD 75-Yr Gridded Data (1951-2025)    │
                                  └────────────────────┬────────────────────┘
                                                       │
                     ┌─────────────────────────────────┼─────────────────────────────────┐
                     ▼                                 ▼                                 ▼
      ┌──────────────────────────────┐  ┌──────────────────────────────┐  ┌──────────────────────────────┐
      │  Oceanic Drivers (ONI / DMI) │  │   ETOPO1 Elevation & Coast   │  │   Bhadali Lunar Features     │
      └──────────────┬───────────────┘  └──────────────┬───────────────┘  └──────────────┬───────────────┘
                     │                                 │                                 │
                     └─────────────────────────────────┼─────────────────────────────────┘
                                                       │
                                                       ▼
                                     ┌───────────────────────────────────┐
                                     │     merged_climate_data_v2.csv    │
                                     └─────────────────┬─────────────────┘
                                                       │
                        ┌──────────────────────────────┴──────────────────────────────┐
                        ▼                                                             ▼
     ┌────────────────────────────────────┐                        ┌────────────────────────────────────┐
     │   LightGBM Temperature Pipeline    │                        │  XGBoost 2-Stage Rainfall Cascade  │
     │  (R² = 98.65%, MAE = 0.49°C)       │                        │  (Classification + Regressor Q10/90│
     └─────────────────┬──────────────────┘                        └─────────────────┬──────────────────┘
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │
                                                      ▼
                                   ┌─────────────────────────────────────┐
                                   │ Real-Time Open-Meteo Inference Engine│
                                   └──────────────────┬──────────────────┘
                                                      │
                                                      ▼
                                   ┌─────────────────────────────────────┐
                                   │   RIT Web Dashboard     │
                                   └─────────────────────────────────────┘
```

1. **Daily Temperature Prediction (LightGBM Regressors)**:
   - High-precision daily maximum and minimum temperature forecasting ($\text{MAE} \le 0.49^\circ\text{C}$, $R^2 = 98.65\%$).
2. **Zero-Inflated Rainfall Forecasting (XGBoost 2-Stage Cascade)**:
   - **Stage 1**: Binary Classifier for precipitation occurrence ($>0.1\text{ mm}$).
   - **Stage 2a**: General Regressor for non-extreme rainfall.
   - **Stage 2b / 3**: Extreme Rainfall Classifier & Regressor with Quantile Uncertainty Bounds ($Q_{10}, Q_{90}$).
3. **Multi-Scale Climate Drivers**:
   - Integrates NOAA Oceanic Niño Index (ONI), Indian Ocean Dipole Dipole Mode Index (DMI), ETOPO1 Digital Elevation Models, and Distance-to-Coast metrics.
4. **Bhadali Folk Astro-Meteorological Validation**:
   - Integrates 27 *Nakshatras* (lunar mansions), *Tithi* phases, and *Paksha* cycles to validate traditional climate indicators.
5. **Real-Time Antecedent Lag Fetcher & Inference Engine**:
   - Queries Open-Meteo API for target grid coordinates to supply real-world observed precipitation/temperature lags.

---

## 📥 Data Sourcing & Download Guide

To train or reproduce the machine learning models from scratch, obtain the official datasets listed below and place them in the project root directory.

### 1. India Meteorological Department (IMD) 75-Year Gridded Datasets (1951–2025)
- **Parameters**: Daily Maximum Temperature ($T_{\text{max}}$), Daily Minimum Temperature ($T_{\text{min}}$), and Daily Rainfall ($R$).
- **Spatial Resolution**: $0.25^\circ \times 0.25^\circ$ (~28 km grid spacing across India).
- **Temporal Coverage**: January 1, 1951 – December 31, 2025.
- **Download Instructions**:
  1. Visit the **IMD Pune Data Center** or **National Data Centre (NDC)** portal: [https://imdpune.gov.in](https://imdpune.gov.in/)
  2. Alternatively, use the official Python library [`imdpy`](https://pypi.org/project/imdpy/) to download netCDF / binary files:
     ```bash
     pip install imdpy
     python -c "import imdpy; imdpy.get_data('rain', 1951, 2025)"
     python -c "import imdpy; imdpy.get_data('tmax', 1951, 2025)"
     python -c "import imdpy; imdpy.get_data('tmin', 1951, 2025)"
     ```
  3. Merge the gridded netCDF files into `merged_climate_data.csv` containing columns: `Latitude`, `Longitude`, `Date`, `Year`, `Month`, `Day`, `Max_Temp`, `Min_Temp`, `Rainfall`.

### 2. NOAA Oceanic Niño Index (ONI - SST Anomalies)
- **Parameters**: Monthly Niño 3.4 SST Anomaly ($^\circ\text{C}$).
- **Download Link**: [NOAA CPC ONI Data](https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt)
- Save raw data or run `py ingest_drivers.py` (which automates fetching and saving to `data/drivers/oni_nino34.csv`).

### 3. NOAA / JAMSTEC Indian Ocean Dipole (IOD DMI Index)
- **Parameters**: Monthly Dipole Mode Index ($^\circ\text{C}$).
- **Download Link**: [NOAA PSL DMI Index](https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.hadisasst.txt)
- Saved automatically by `ingest_drivers.py` to `data/drivers/iod_dmi.csv`.

### 4. ETOPO1 Global Relief Digital Elevation Model (DEM)
- **Parameters**: Altitude / Elevation in meters ($m$) and distance to coast ($km$).
- **Download Link**: [NOAA NCEI ETOPO1](https://www.ngdc.noaa.gov/mgg/global/global.html)
- Processed into `data/drivers/elevation_grid.csv` and `data/drivers/coast_distance_grid.csv`.

---

## 🛠️ Installation & Quickstart

### Prerequisites
- **Python**: Version 3.10 or higher
- **Node.js**: Version 18 or higher (for serving the frontend dashboard)

### 1. Clone the Repository
```bash
git clone https://github.com/
cd RIT-Digital-Twin
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

> **Required Packages**: `lightgbm`, `xgboost`, `scikit-learn`, `pandas`, `numpy`, `requests`, `ephem`, `matplotlib`, `seaborn`

---

## 🚀 Running the Full Pipeline

### Step 1: Ingest Climate Drivers & Generate Augmented Dataset
Merges NOAA ONI, IOD DMI, Elevation, Coastal Distance, and ENSO/IOD phases into `merged_climate_data_v2.csv`:
```bash
py ingest_drivers.py
```

### Step 2: Build Climatology Lookup Tables
Generates 75-year historical monthly and weekly averages for instant inference lookup:
```bash
py build_climatology_table.py
```

### Step 3: Train Temperature AI Models
Trains LightGBM models for $T_{\text{max}}$ and $T_{\text{min}}$, producing `max_temp_model.pkl` and `min_temp_model.pkl`:
```bash
py temperature_model.py
```

### Step 4: Train Rainfall Cascade AI Models
Trains the 2-Stage XGBoost cascade and quantile regressors ($Q_{10}, Q_{90}$):
```bash
py rainfall_model.py
```

### Step 5: Run Terminal Inference Engine (e.g. Ahmedabad)
Executes real-time inference for any target coordinate and date, fetching antecedent weather from Open-Meteo:
```bash
py run_inference.py --lat 23.03 --lon 72.58 --date 2026-07-29
```

### Step 6: Launch Interactive Web Dashboard
Serve the frontend application:
```bash
npx -y serve frontend -l 8765
```
Open **[http://localhost:8765](http://localhost:8765)** in your browser.

---

## 📊 Model Performance Benchmarks

### Daily Temperature Models (LightGBM)
| Target Metric | $R^2$ Score | Mean Absolute Error ($\text{MAE}$) | Root Mean Square Error ($\text{RMSE}$) |
| :--- | :--- | :--- | :--- |
| **Max Temperature ($T_{\text{max}}$)** | **98.65%** | **0.49 °C** | **0.68 °C** |
| **Min Temperature ($T_{\text{min}}$)** | **97.82%** | **0.42 °C** | **0.59 °C** |

### Live City Validation (July 29, 2026 vs Open-Meteo Recorded Data)
| City | Predicted Max Temp | Actual Recorded Max | Max Error | Predicted Min Temp | Actual Recorded Min | Min Error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Chennai** | **30.51 °C** | 30.40 °C | **+0.11 °C** | **28.00 °C** | 28.30 °C | **-0.30 °C** |
| **Bengaluru** | **29.14 °C** | 29.30 °C | **-0.16 °C** | **22.10 °C** | 22.70 °C | **-0.60 °C** |
| **Kolkata** | **31.38 °C** | 30.80 °C | **+0.58 °C** | **25.95 °C** | 25.40 °C | **+0.55 °C** |
| **Ahmedabad** | **31.74 °C** | 32.30 °C | **-0.56 °C** | **26.16 °C** | 26.70 °C | **-0.54 °C** |

---

## 📂 Repository Directory Structure

```text
RIT-Digital-Twin/
├── frontend/                        # RIT Web Dashboard
│   ├── index.html                   # Core Application UI Layout
│   ├── styles.css                   # Dark-mode Glassmorphism CSS Design Tokens
│   ├── app.js                       # Main Application Controller & Bootstram
│   ├── data.js                      # Real Model Data Loaders & City Offsets
│   ├── map.js                       # Leaflet Choropleth Map Engine
│   ├── charts.js                    # Chart.js Interactive Time-Series Graphs
│   ├── whatif.js                    # What-If Scenario Matrix Engine
│   ├── pages.js                     # Page Router & Navigation Controller
│   └── data/                        # Live Synced Prediction Payload JSONs
├── ingest_drivers.py                # Driver Ingestion (ONI, DMI, DEM, Coast)
├── build_climatology_table.py       # 75-Yr Climatology Normal Generator
├── temperature_model.py             # LightGBM Temperature Model Training
├── rainfall_model.py                # 2-Stage XGBoost Cascade & Quantile Regressors
├── bhadali_features.py              # Astro-Meteorological Nakshatra Engine
├── run_inference.py                 # Real-Time Open-Meteo Inference Script
├── generate_all_city_predictions.py  # Multi-City Batch Inference Auto-Syncer
├── sync_frontend_data.py            # Automated Frontend Artifact Sync Engine
├── README.md                        # Documentation
└── .gitignore                       # Large Dataset Exclusions
```

---

## 📜 License & Acknowledgments

- **License**: Released under the [MIT License](LICENSE).
- **Data Acknowledgments**: India Meteorological Department (IMD), NOAA Climate Prediction Center, Open-Meteo API.
