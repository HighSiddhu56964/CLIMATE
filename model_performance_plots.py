"""
================================================================================
  Model Performance Visualization Suite
  Climate Digital Twin — RIT / ISRO BAH 2026
================================================================================

Generates a LinkedIn-carousel-ready set of PNGs proving model performance:
  • Temperature regression  (LightGBM)
  • Rainfall classification (XGBoost cascade)
  • Spatial / trend diagnostics

Loads pre-trained .pkl models and replays the exact feature engineering from
temperature_model.py and rainfall_model.py so predictions match what the models
were trained on.

Usage:
    Full run (final output):
        py model_performance_plots.py

    Dev / debug iteration (fast, 10 % of grid points):
        py model_performance_plots.py --sample-frac 0.10

Output:
    All PNGs saved to ./useful_plots/ at 300 DPI, dark ISRO mission-control theme.
    Spatial error map saved as both HTML and PNG (via kaleido).
================================================================================
"""

import argparse
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import pickle
import json
import os
import gc
import warnings
import time
from scipy.signal import lfilter
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    accuracy_score, f1_score,
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# THEME — dark glassmorphism / ISRO mission-control (matches Streamlit dashboard)
# ═══════════════════════════════════════════════════════════════════════════════
BG_COLOR      = "#0b0f19"
PANEL_COLOR   = "#111827"
GRID_COLOR    = "#1f2937"
TEXT_COLOR    = "#e5e7eb"
ACCENT_CYAN   = "#22d3ee"
ACCENT_TEAL   = "#2dd4bf"
ACCENT_AMBER  = "#f59e0b"
ACCENT_ROSE   = "#fb7185"
ACCENT_VIOLET = "#a78bfa"

mpl.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": PANEL_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.4,
    "font.size": 11,
    "font.family": "sans-serif",
    "savefig.facecolor": BG_COLOR,
    "savefig.dpi": 300,
    "legend.facecolor": PANEL_COLOR,
    "legend.edgecolor": GRID_COLOR,
})

OUT_DIR = "useful_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Model Performance Visualization Suite")
parser.add_argument(
    "--sample-frac", type=float, default=1.0,
    help="Fraction of grid points to use (0.0-1.0). Use 0.10-0.20 for fast dev "
         "iteration; 1.0 for final output. Default: 1.0"
)
args = parser.parse_args()
SAMPLE_FRAC = max(0.01, min(1.0, args.sample_frac))

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (feature lists matching the trained models exactly)
# ═══════════════════════════════════════════════════════════════════════════════

# Temperature model features (37 features — LightGBM Booster)
TEMP_FEATURES = [
    "Latitude", "Longitude",
    "Year", "Month", "Day", "DayOfYear", "Season_Code",
    "Month_sin", "Month_cos", "Day_sin", "Day_cos",
    "MaxTemp_lag1", "MaxTemp_lag3", "MaxTemp_lag7",
    "MinTemp_lag1", "MinTemp_lag3", "MinTemp_lag7",
    "Rainfall_lag1",
    "MaxTemp_roll7", "MaxTemp_roll30",
    "MinTemp_roll7", "MinTemp_roll30",
    "Rain_roll7",
    "Clim_MaxTemp", "Clim_MinTemp",
    "Diurnal_Range", "Rainfall",
    "ONI", "DMI", "Elevation_m", "Dist_Coast_km", "Log_Dist_Coast",
    "ENSO_Phase", "IOD_Phase",
    "ONI_x_Monsoon", "DMI_x_Monsoon", "Elevation_x_Monsoon",
]

# Bhadali Vakyo feature names (for highlighting in plot 7)
BHADALI_FEATURE_NAMES = [
    "Moon_Phase_Angle", "Moon_Phase_Sin", "Moon_Phase_Cos", "Moon_Illumination",
    "Tithi", "Tithi_Sin", "Tithi_Cos", "Paksha", "Nakshatra_Sin", "Nakshatra_Cos",
    "Lunar_Month", "Vara", "Is_Swati", "Is_Rohini", "Is_Anuradha", "Is_Hasta",
    "Is_Shravana", "Is_Ardra", "Is_Purnima", "Is_Amavas", "Is_Saptami",
    "Bhadali_Score", "Swati_x_Monsoon", "Rohini_x_Paksha", "Purnima_x_Monsoon",
]

NEW_FEATURE_NAMES = [
    "API", "Clim_Rainfall_Week", "Clim_Rain_Prob_Week", "Monsoon_Progress_Days",
    "Lat_x_DayOfYear", "Lon_x_DayOfYear", "Weeks_Since_Peak_Rain", "Rainfall_Anom_roll7",
    "Neighbor_Rain_Mean_roll7", "Temp_Anomaly", "Humidity_Proxy", "Humidity_Anomaly",
    "Pressure_Anomaly", "Cloud_Top_Temp", "Moisture_Transport", "Convergence_850hPa",
]

# LightGBM parameters (identical to temperature_model.py)
LGB_PARAMS = {
    "objective":        "regression",
    "metric":           "rmse",
    "n_estimators":     1000,
    "learning_rate":    0.05,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "n_jobs":           -1,
    "verbose":          -1,
    "random_state":     42,
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _style_axes(ax, title, xlabel, ylabel):
    """Apply dark-theme styling to a matplotlib Axes object."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    return ax


def _dry_spell_vec(series):
    """Vectorised dry-spell counter per grid cell."""
    shifted = series.shift(1)
    is_dry = (shifted <= 0.1).astype(int)
    not_dry = (is_dry == 0).cumsum()
    return is_dry.groupby(not_dry).cumsum().astype(np.float32)


def _wet_spell_vec(series):
    """Vectorised wet-spell counter per grid cell."""
    shifted = series.shift(1)
    is_wet = (shifted > 0.1).astype(int)
    not_wet = (is_wet == 0).cumsum()
    return is_wet.groupby(not_wet).cumsum().astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  Model Performance Visualization Suite")
print("  Climate Digital Twin — RIT / ISRO BAH 2026")
print("=" * 70)
print(f"\n  --sample-frac = {SAMPLE_FRAC}")

t_start = time.time()

print("\n" + "=" * 70)
print("STEP 1: Loading merged_climate_data_v2.csv ...")
print("=" * 70)

dtypes_climate = {
    "Year": "int16", "Month": "int8", "Day": "int8",
    "Season": "category", "Latitude": "float32", "Longitude": "float32",
    "Max_Temp": "float32", "Min_Temp": "float32", "Diurnal_Range": "float32",
    "Rainfall": "float32", "ONI": "float32", "DMI": "float32",
    "Elevation_m": "float32", "Dist_Coast_km": "float32", "Log_Dist_Coast": "float32",
    "ENSO_Phase": "int8", "IOD_Phase": "int8",
    "ONI_x_Monsoon": "float32", "DMI_x_Monsoon": "float32",
    "Elevation_x_Monsoon": "float32",
}
df = pd.read_csv("merged_climate_data_v2.csv", parse_dates=["Date"], dtype=dtypes_climate)
df["Rainfall"] = df["Rainfall"].fillna(0)
print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

# ── Optional grid-point sampling ──
if SAMPLE_FRAC < 1.0:
    all_grids = df[["Latitude", "Longitude"]].drop_duplicates()
    n_sample = max(10, int(len(all_grids) * SAMPLE_FRAC))
    rng = np.random.default_rng(42)
    sampled = all_grids.sample(n=n_sample, random_state=42)
    # Always include Ahmedabad (23.5, 72.5) for Plot 4
    abad = all_grids[(all_grids["Latitude"] == 23.5) & (all_grids["Longitude"] == 72.5)]
    sampled = pd.concat([sampled, abad]).drop_duplicates()
    df = df.merge(sampled, on=["Latitude", "Longitude"], how="inner")
    print(f"  Sampled to {len(sampled)} grid points -> {len(df):,} rows")
    gc.collect()

# ── Load Bhadali features ──
BHADALI_CSV = "bhadali_features.csv"
has_bhadali = os.path.exists(BHADALI_CSV)
if has_bhadali:
    print("  Loading Bhadali features ...")
    df_bhadali = pd.read_csv(BHADALI_CSV)
    df_bhadali["Date"] = pd.to_datetime(df_bhadali["Date"], format="mixed")
    bhadali_cols = [c for c in df_bhadali.columns if c not in ["Date", "Year", "Month", "Day"]]
    for col in bhadali_cols:
        if df_bhadali[col].dtype == "float64":
            df_bhadali[col] = df_bhadali[col].astype(np.float32)
        elif df_bhadali[col].dtype == "int64":
            df_bhadali[col] = df_bhadali[col].astype(np.int8)
    df = df.merge(df_bhadali[["Date"] + bhadali_cols], on="Date", how="left")
    del df_bhadali
    gc.collect()
    print(f"  Merged Bhadali features ({len(bhadali_cols)} columns)")

print(f"  Data load time: {time.time() - t_start:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING  (replays temperature_model.py + rainfall_model.py)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Feature Engineering ...")
print("=" * 70)
t_feat = time.time()

# ── Sort by location + date (required for correct lag/rolling) ──
df = df.sort_values(["Latitude", "Longitude", "Date"]).reset_index(drop=True)

# ── 2a. Shared base features ──
season_map = {"Winter": 0, "Pre-Monsoon": 1, "Monsoon": 2, "Post-Monsoon": 3}
df["Season_Code"] = df["Season"].map(season_map)
df["Month_sin"] = np.sin(2 * np.pi * df["Month"].astype(np.float32) / 12).astype(np.float32)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"].astype(np.float32) / 12).astype(np.float32)
df["DayOfYear"] = df["Date"].dt.dayofyear.astype(np.int16)
doy_f = df["DayOfYear"].astype(np.float32)
df["Day_sin"] = np.sin(2 * np.pi * doy_f / 365.0).astype(np.float32)
df["Day_cos"] = np.cos(2 * np.pi * doy_f / 365.0).astype(np.float32)
del doy_f
df["Is_Monsoon"] = df["Month"].isin([6, 7, 8, 9]).astype(np.int8)
df["Lat_Zone"] = pd.cut(df["Latitude"], bins=[0, 15, 20, 25, 40], labels=[0, 1, 2, 3]).astype(np.float32)
df["Week"] = df["Date"].dt.isocalendar().week.astype(np.int8)
print("  Base features done")

# ── 2b. Lag features ──
print("  Computing lag features ...")
grp = df.groupby(["Latitude", "Longitude"])

# Temperature lags (for temperature model)
df["MaxTemp_lag1"] = grp["Max_Temp"].shift(1)
df["MaxTemp_lag3"] = grp["Max_Temp"].shift(3)
df["MaxTemp_lag7"] = grp["Max_Temp"].shift(7)
df["MinTemp_lag1"] = grp["Min_Temp"].shift(1)
df["MinTemp_lag3"] = grp["Min_Temp"].shift(3)
df["MinTemp_lag7"] = grp["Min_Temp"].shift(7)
df["Rainfall_lag1"] = grp["Rainfall"].shift(1)

# Rainfall lags (for rainfall model)
for lag in [1, 2, 3, 7, 14]:
    df[f"Rain_lag{lag}"] = grp["Rainfall"].shift(lag).astype(np.float32)
df["Rain_lag1_binary"] = (df["Rain_lag1"] > 0.1).astype(np.float32)

# ── 2c. Rolling features ──
print("  Computing rolling features ...")

# Temperature rolling
df["MaxTemp_roll7"]  = grp["Max_Temp"].transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
df["MaxTemp_roll30"] = grp["Max_Temp"].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
df["MinTemp_roll7"]  = grp["Min_Temp"].transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
df["MinTemp_roll30"] = grp["Min_Temp"].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
df["Rain_roll7"]     = grp["Rainfall"].transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())

# Rainfall rolling (sums, not means — rainfall_model.py uses .sum())
df["Rain_roll3"]  = grp["Rainfall"].transform(lambda x: x.shift(1).rolling(3,  min_periods=1).sum()).astype(np.float32)
# Overwrite Rain_roll7 for rainfall model (sum, not mean) — keep mean version for temp
df["Rain_roll7_sum"]  = grp["Rainfall"].transform(lambda x: x.shift(1).rolling(7,  min_periods=1).sum()).astype(np.float32)
df["Rain_roll14"] = grp["Rainfall"].transform(lambda x: x.shift(1).rolling(14, min_periods=1).sum()).astype(np.float32)
df["Rain_roll30"] = grp["Rainfall"].transform(lambda x: x.shift(1).rolling(30, min_periods=1).sum()).astype(np.float32)
df["Rain_days7"]  = grp["Rainfall"].transform(lambda x: (x.shift(1) > 0.1).rolling(7, min_periods=1).sum()).astype(np.float32)
df["Rain_max7"]   = grp["Rainfall"].transform(lambda x: x.shift(1).rolling(7,  min_periods=1).max()).astype(np.float32)

# ── 2d. Climatology (temperature: all years; rainfall: train only) ──
print("  Computing climatology ...")

# Temperature climatology (all years, matches temperature_model.py)
clim_temp = df.groupby(["Latitude", "Longitude", "Month"])[["Max_Temp", "Min_Temp"]].mean()
clim_temp.columns = ["Clim_MaxTemp", "Clim_MinTemp"]
clim_temp = clim_temp.reset_index()
df = df.merge(clim_temp, on=["Latitude", "Longitude", "Month"], how="left")
del clim_temp

# Rainfall climatology (train only, Year <= 2018, matches rainfall_model.py)
train_df_raw = df[df["Year"] <= 2018]

clim_r = train_df_raw.groupby(["Latitude", "Longitude", "Month"])["Rainfall"].mean().reset_index().rename(columns={"Rainfall": "Clim_Rainfall"})
clim_p = train_df_raw.groupby(["Latitude", "Longitude", "Month"]).apply(
    lambda x: (x["Rainfall"] > 0.1).mean(), include_groups=False
).reset_index().rename(columns={0: "Clim_Rain_Prob"})
df = df.merge(clim_r, on=["Latitude", "Longitude", "Month"], how="left")
df = df.merge(clim_p, on=["Latitude", "Longitude", "Month"], how="left")

# Dry season probability
clim_dry = train_df_raw.groupby(["Latitude", "Longitude", "Month"]).apply(
    lambda x: (x["Rainfall"] <= 0.1).mean(), include_groups=False
).reset_index().rename(columns={0: "Dry_Season_Prob"})
df = df.merge(clim_dry, on=["Latitude", "Longitude", "Month"], how="left")

for col in ["Clim_Rainfall", "Clim_Rain_Prob"]:
    df[col] = df[col].fillna(0.0).astype(np.float32)
df["Dry_Season_Prob"] = df["Dry_Season_Prob"].fillna(1.0).astype(np.float32)

# ── 2e. Dry/Wet spells ──
print("  Computing dry/wet spells ...")
df["Dry_Spell"] = grp["Rainfall"].transform(_dry_spell_vec)
df["Wet_Spell"] = grp["Rainfall"].transform(_wet_spell_vec)
df["Dry_Spell_x_Monsoon"] = (df["Dry_Spell"] * df["Is_Monsoon"]).astype(np.float32)
df["Dry_Spell_x_NotMonsoon"] = (df["Dry_Spell"] * (1 - df["Is_Monsoon"])).astype(np.float32)

# ── 2f. Spatial neighbor features ──
print("  Computing spatial neighbor features ...")
df["Date_next"] = df["Date"] + pd.Timedelta(days=1)
yesterday_lookup = df[["Date", "Latitude", "Longitude", "Rainfall"]].copy()
yesterday_lookup.columns = ["Date_next", "Latitude", "Longitude", "Yday_Rain"]

for direction, dlat, dlon in [("N", -1.0, 0.0), ("S", 1.0, 0.0), ("E", 0.0, -1.0), ("W", 0.0, 1.0)]:
    n = yesterday_lookup.copy()
    n["Latitude"]  = n["Latitude"]  - dlat
    n["Longitude"] = n["Longitude"] - dlon
    df = df.merge(n.rename(columns={"Yday_Rain": f"Rain_{direction}"}),
                  on=["Date_next", "Latitude", "Longitude"], how="left")
    del n

df.drop(columns=["Date_next"], inplace=True)
for col in ["Rain_N", "Rain_S", "Rain_E", "Rain_W"]:
    df[col] = df[col].fillna(0.0).astype(np.float32)
df["Neighbor_Rain_Mean"] = ((df["Rain_N"] + df["Rain_S"] + df["Rain_E"] + df["Rain_W"]) / 4.0).astype(np.float32)
df["Neighbor_Rain_Max"]  = df[["Rain_N", "Rain_S", "Rain_E", "Rain_W"]].max(axis=1).astype(np.float32)
df["Neighbor_Any_Rain"]  = (df["Neighbor_Rain_Mean"] > 0.1).astype(np.float32)
df["Neighbor_Rain_Mean_roll7"] = df.groupby(["Latitude", "Longitude"])["Neighbor_Rain_Mean"].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
).astype(np.float32)
df.drop(columns=["Rain_N", "Rain_S", "Rain_E", "Rain_W"], inplace=True)
del yesterday_lookup
gc.collect()

# ── 2g. API (Antecedent Precipitation Index) ──
print("  Computing API ...")
df["API"] = df.groupby(["Latitude", "Longitude"])["Rain_lag1"].transform(
    lambda x: lfilter([1.0], [1.0, -0.85], x.fillna(0.0))
).astype(np.float32)

# ── 2h. Weekly climatology + monsoon progression ──
print("  Computing weekly climatology ...")

clim_r_w = train_df_raw.groupby(["Latitude", "Longitude", "Week"])["Rainfall"].mean().reset_index().rename(columns={"Rainfall": "Clim_Rainfall_Week"})
clim_p_w = train_df_raw.groupby(["Latitude", "Longitude", "Week"]).apply(
    lambda x: (x["Rainfall"] > 0.1).mean(), include_groups=False
).reset_index().rename(columns={0: "Clim_Rain_Prob_Week"})
df = df.merge(clim_r_w, on=["Latitude", "Longitude", "Week"], how="left")
df = df.merge(clim_p_w, on=["Latitude", "Longitude", "Week"], how="left")
for col in ["Clim_Rainfall_Week", "Clim_Rain_Prob_Week"]:
    df[col] = df[col].fillna(0.0).astype(np.float32)

# Monsoon progression
df["Monsoon_Progress_Days"] = np.where(
    df["Month"].isin([6, 7, 8, 9]), (df["DayOfYear"] - 152).astype(np.float32), 0.0
).astype(np.float32)
df["Lat_x_DayOfYear"] = (df["Latitude"] * df["DayOfYear"]).astype(np.float32)
df["Lon_x_DayOfYear"] = (df["Longitude"] * df["DayOfYear"]).astype(np.float32)

# Peak rain week
if os.path.exists("peak_rain_week.csv"):
    peak_week = pd.read_csv("peak_rain_week.csv")
    df = df.merge(peak_week[["Latitude", "Longitude", "Peak_Rain_Week"]], on=["Latitude", "Longitude"], how="left")
else:
    week_avg = train_df_raw.groupby(["Latitude", "Longitude", "Week"])["Rainfall"].mean().fillna(0.0).reset_index()
    idx_max = week_avg.groupby(["Latitude", "Longitude"])["Rainfall"].idxmax()
    peak_week = week_avg.loc[idx_max, ["Latitude", "Longitude", "Week"]].rename(columns={"Week": "Peak_Rain_Week"})
    df = df.merge(peak_week, on=["Latitude", "Longitude"], how="left")

df["Peak_Rain_Week"] = df["Peak_Rain_Week"].fillna(28).astype(np.int8)
df["Weeks_Since_Peak_Rain"] = (df["Week"] - df["Peak_Rain_Week"]).astype(np.float32)
df["Rainfall_Anom_roll7"] = (df["Rain_roll7_sum"] - (df["Clim_Rainfall_Week"] * 7.0)).astype(np.float32)

# ── 2i. Atmospheric diagnostic proxies ──
print("  Computing atmospheric proxies ...")
clim_temp_w = train_df_raw.groupby(["Latitude", "Longitude", "Week"])[["Max_Temp", "Min_Temp"]].mean().reset_index()
clim_temp_w.columns = ["Latitude", "Longitude", "Week", "Clim_Max_Temp", "Clim_Min_Temp"]
df = df.merge(clim_temp_w, on=["Latitude", "Longitude", "Week"], how="left")
df["Clim_Max_Temp"] = df["Clim_Max_Temp"].fillna(df["Max_Temp"].mean()).astype(np.float32)
df["Clim_Min_Temp"] = df["Clim_Min_Temp"].fillna(df["Min_Temp"].mean()).astype(np.float32)

df["Temp_Anomaly"] = (df["Max_Temp"] - df["Clim_Max_Temp"]).astype(np.float32)
df["Humidity_Proxy"] = (100.0 - 5.0 * df["Diurnal_Range"]).clip(10.0, 100.0).astype(np.float32)

clim_hum = (100.0 - 5.0 * train_df_raw.groupby(["Latitude", "Longitude", "Week"])["Diurnal_Range"].mean()).reset_index()
clim_hum.columns = ["Latitude", "Longitude", "Week", "Clim_Humidity_Proxy"]
df = df.merge(clim_hum, on=["Latitude", "Longitude", "Week"], how="left")
df["Clim_Humidity_Proxy"] = df["Clim_Humidity_Proxy"].fillna(df["Humidity_Proxy"].mean()).astype(np.float32)
df["Humidity_Anomaly"] = (df["Humidity_Proxy"] - df["Clim_Humidity_Proxy"]).astype(np.float32)

df["Pressure_Anomaly"] = (-0.5 * df["Temp_Anomaly"] - 0.2 * df["Rain_roll3"]).clip(-15.0, 15.0).astype(np.float32)
df["Cloud_Top_Temp"]   = (295.0 - 15.0 * (df["Rain_lag1"] > 0.1) - 5.0 * df["Rain_roll3"] - 0.1 * df["Humidity_Proxy"]).clip(200.0, 310.0).astype(np.float32)
df["Moisture_Transport"] = (((1.5 * df["Is_Monsoon"] + 0.5) * df["Humidity_Proxy"]) + 0.3 * df["Rain_roll7_sum"]).clip(0.0, 200.0).astype(np.float32)
df["Convergence_850hPa"] = (-0.8 * df["Pressure_Anomaly"] + 0.3 * df["Neighbor_Rain_Mean"]).clip(-20.0, 20.0).astype(np.float32)

del train_df_raw, clim_r, clim_p, clim_dry, clim_r_w, clim_p_w, clim_temp_w, clim_hum
gc.collect()

# Rename Rain_roll7_sum -> Rain_roll7 for the rainfall model (it expects "Rain_roll7" as sum)
# The temperature model uses Rain_roll7 as mean (already set). We keep both.
# The rainfall feature list says "Rain_roll7" which in rainfall_model.py is a SUM.
# We need to swap: rename the sum version to what the rainfall model expects.
# Save the mean version under a temp name first.
df["Rain_roll7_mean"] = df["Rain_roll7"]  # temp model uses mean
df["Rain_roll7"] = df["Rain_roll7_sum"]   # rainfall model uses sum
df.drop(columns=["Rain_roll7_sum"], inplace=True)

print(f"  Feature engineering complete: {len(df.columns)} columns, {time.time() - t_feat:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: DATA SPLITTING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Splitting into Train / Val / Test ...")
print("=" * 70)

# Clean for temperature targets
df.dropna(subset=["Max_Temp", "Min_Temp"], inplace=True)
df = df[(df["Max_Temp"] >= -20) & (df["Max_Temp"] <= 55)]
df = df[(df["Min_Temp"] >= -20) & (df["Min_Temp"] <= 45)]
df = df[df["Max_Temp"] >= df["Min_Temp"]]

year_col = df["Year"]
train_mask = year_col <= 2018
val_mask   = (year_col >= 2019) & (year_col <= 2021)
test_mask  = year_col >= 2022

print(f"  Train : {train_mask.sum():,} rows (<=2018)")
print(f"  Val   : {val_mask.sum():,} rows  (2019-2021)")
print(f"  Test  : {test_mask.sum():,} rows  (>=2022)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: LOAD MODELS & PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Loading models & generating predictions ...")
print("=" * 70)

# ── 4a. Temperature Models ──
with open("max_temp_model.pkl", "rb") as f:
    max_temp_model = pickle.load(f)
with open("min_temp_model.pkl", "rb") as f:
    min_temp_model = pickle.load(f)
with open("feature_columns.pkl", "rb") as f:
    temp_feature_cols = pickle.load(f)
print(f"  Loaded temperature models ({max_temp_model.num_feature()} features, best_iter={max_temp_model.best_iteration})")

# Temperature model uses Rain_roll7 as mean
# Temporarily swap back for temperature predictions
df["Rain_roll7_rain"] = df["Rain_roll7"]  # save rainfall version
df["Rain_roll7"] = df["Rain_roll7_mean"]  # restore mean version for temp

# Filter to rows with all temperature features available
temp_available = df[temp_feature_cols].dropna()
temp_test_idx = temp_available.index[temp_available.index.isin(df.index[test_mask])]
temp_train_idx = temp_available.index[temp_available.index.isin(df.index[train_mask])]

X_test_temp  = df.loc[temp_test_idx, temp_feature_cols]
y_max_test   = df.loc[temp_test_idx, "Max_Temp"]
y_min_test   = df.loc[temp_test_idx, "Min_Temp"]

print(f"  Temperature test set: {len(X_test_temp):,} rows")

max_preds = max_temp_model.predict(X_test_temp, num_iteration=max_temp_model.best_iteration)
min_preds = min_temp_model.predict(X_test_temp, num_iteration=min_temp_model.best_iteration)
print(f"  Temperature predictions done.")

# Restore rainfall version
df["Rain_roll7"] = df["Rain_roll7_rain"]
df.drop(columns=["Rain_roll7_rain"], inplace=True)

# ── 4b. Rainfall Model ──
with open("rainfall_feature_cols.pkl", "rb") as f:
    rain_feature_cols = pickle.load(f)
with open("rainfall_classifier.pkl", "rb") as f:
    rain_classifier = pickle.load(f)
print(f"  Loaded rainfall classifier ({rain_classifier.n_features_in_} features)")

# Build rain binary target
df["Rain_Binary"] = (df["Rainfall"] > 0.1).astype(np.int8)

# Filter to rows with all rainfall features available
rain_available = df[rain_feature_cols].dropna()
rain_test_idx = rain_available.index[rain_available.index.isin(df.index[test_mask])]

X_test_rain  = df.loc[rain_test_idx, rain_feature_cols]
y_rain_test  = df.loc[rain_test_idx, "Rain_Binary"]
yr_rain_test = df.loc[rain_test_idx, "Rainfall"]

print(f"  Rainfall test set: {len(X_test_rain):,} rows")

rain_probs = rain_classifier.predict_proba(X_test_rain)[:, 1]
rain_preds_binary = (rain_probs >= 0.5).astype(int)
print(f"  Rainfall predictions done.")

# Compute metrics
max_mae  = mean_absolute_error(y_max_test, max_preds)
max_rmse = np.sqrt(mean_squared_error(y_max_test, max_preds))
max_r2   = r2_score(y_max_test, max_preds)
min_mae  = mean_absolute_error(y_min_test, min_preds)
min_rmse = np.sqrt(mean_squared_error(y_min_test, min_preds))
min_r2   = r2_score(y_min_test, min_preds)

rain_acc = accuracy_score(y_rain_test, rain_preds_binary)
rain_f1  = f1_score(y_rain_test, rain_preds_binary)

print(f"\n  Max Temp — MAE: {max_mae:.3f}°C  RMSE: {max_rmse:.3f}°C  R²: {max_r2:.4f}")
print(f"  Min Temp — MAE: {min_mae:.3f}°C  RMSE: {min_rmse:.3f}°C  R²: {min_r2:.4f}")
print(f"  Rainfall — Acc: {rain_acc:.4f}  F1: {rain_f1:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: RETRAIN WITH EVAL TRACKING (Plot 1)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Training curve — retrain with eval tracking ...")
print("=" * 70)

import lightgbm as lgb

EVALS_CACHE = os.path.join(OUT_DIR, "evals_result_cache.json")

if os.path.exists(EVALS_CACHE):
    print("  Loading cached training curve from previous run ...")
    with open(EVALS_CACHE, "r") as f:
        evals_cached = json.load(f)
    evals_temp = evals_cached.get("temperature", {})
    evals_rain = evals_cached.get("rainfall", {})
else:
    print("  No cache found — retraining with record_evaluation() ...")

    # Subsample train for speed (15% — captures genuine curve shape)
    RETRAIN_FRAC = 0.15
    rng_rt = np.random.default_rng(42)

    # --- Temperature retrain ---
    df["Rain_roll7_rain2"] = df["Rain_roll7"]
    df["Rain_roll7"] = df["Rain_roll7_mean"]

    temp_train_all = df.loc[temp_train_idx, temp_feature_cols]
    y_max_train_all = df.loc[temp_train_idx, "Max_Temp"]

    sub_n = int(len(temp_train_all) * RETRAIN_FRAC)
    sub_idx = rng_rt.choice(len(temp_train_all), size=sub_n, replace=False)
    X_rt_train = temp_train_all.iloc[sub_idx]
    y_rt_train = y_max_train_all.iloc[sub_idx]

    # Validation set for training curve
    temp_val_idx = temp_available.index[temp_available.index.isin(df.index[val_mask])]
    X_rt_val = df.loc[temp_val_idx, temp_feature_cols]
    y_rt_val = df.loc[temp_val_idx, "Max_Temp"]

    # Subsample val for speed
    val_sub_n = min(len(X_rt_val), 200_000)
    val_sub_idx = rng_rt.choice(len(X_rt_val), size=val_sub_n, replace=False)
    X_rt_val = X_rt_val.iloc[val_sub_idx]
    y_rt_val = y_rt_val.iloc[val_sub_idx]

    print(f"  Temp retrain: {len(X_rt_train):,} train, {len(X_rt_val):,} val rows")

    lgb_train_set = lgb.Dataset(X_rt_train, label=y_rt_train)
    lgb_val_set   = lgb.Dataset(X_rt_val,   label=y_rt_val, reference=lgb_train_set)

    evals_result_temp = {}
    lgb_retrain_params = {
        "objective": "regression", "metric": "rmse",
        "num_leaves": 63, "learning_rate": 0.05,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "random_state": 42,
    }

    t_retrain = time.time()
    lgb.train(
        lgb_retrain_params,
        lgb_train_set,
        num_boost_round=1000,
        valid_sets=[lgb_train_set, lgb_val_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.record_evaluation(evals_result_temp),
        ],
    )
    print(f"  Temp retrain done: {time.time() - t_retrain:.1f}s, "
          f"{len(evals_result_temp.get('train', {}).get('rmse', []))} rounds")

    evals_temp = {
        "train_rmse": evals_result_temp.get("train", {}).get("rmse", []),
        "valid_rmse": evals_result_temp.get("valid", {}).get("rmse", []),
    }

    del lgb_train_set, lgb_val_set, X_rt_train, y_rt_train, X_rt_val, y_rt_val
    gc.collect()

    # Restore rain_roll7
    df["Rain_roll7"] = df["Rain_roll7_rain2"]
    df.drop(columns=["Rain_roll7_rain2"], inplace=True)

    # --- Rainfall classifier retrain ---
    import xgboost as xgb

    rain_train_idx = rain_available.index[rain_available.index.isin(df.index[train_mask])]
    rain_val_idx   = rain_available.index[rain_available.index.isin(df.index[val_mask])]

    X_rt_rain_train = df.loc[rain_train_idx, rain_feature_cols]
    y_rt_rain_train = df.loc[rain_train_idx, "Rain_Binary"]
    X_rt_rain_val   = df.loc[rain_val_idx, rain_feature_cols]
    y_rt_rain_val   = df.loc[rain_val_idx, "Rain_Binary"]

    # Subsample
    sub_n_r = int(len(X_rt_rain_train) * RETRAIN_FRAC)
    sub_idx_r = rng_rt.choice(len(X_rt_rain_train), size=sub_n_r, replace=False)
    X_rt_rain_train = X_rt_rain_train.iloc[sub_idx_r]
    y_rt_rain_train = y_rt_rain_train.iloc[sub_idx_r]

    val_sub_n_r = min(len(X_rt_rain_val), 200_000)
    val_sub_idx_r = rng_rt.choice(len(X_rt_rain_val), size=val_sub_n_r, replace=False)
    X_rt_rain_val = X_rt_rain_val.iloc[val_sub_idx_r]
    y_rt_rain_val = y_rt_rain_val.iloc[val_sub_idx_r]

    spw_rt = float((y_rt_rain_train == 0).sum() / (y_rt_rain_train == 1).sum())
    print(f"  Rain retrain: {len(X_rt_rain_train):,} train, {len(X_rt_rain_val):,} val rows")

    t_retrain_r = time.time()
    xgb_retrain = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=500,
        learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw_rt, tree_method="hist", n_jobs=-1, random_state=42,
        early_stopping_rounds=50,
    )
    xgb_retrain.fit(
        X_rt_rain_train, y_rt_rain_train,
        eval_set=[(X_rt_rain_train, y_rt_rain_train), (X_rt_rain_val, y_rt_rain_val)],
        verbose=False,
    )
    xgb_evals = xgb_retrain.evals_result()
    print(f"  Rain retrain done: {time.time() - t_retrain_r:.1f}s, "
          f"{len(xgb_evals.get('validation_0', {}).get('logloss', []))} rounds")

    evals_rain = {
        "train_logloss": xgb_evals.get("validation_0", {}).get("logloss", []),
        "valid_logloss": xgb_evals.get("validation_1", {}).get("logloss", []),
    }

    del xgb_retrain, X_rt_rain_train, y_rt_rain_train, X_rt_rain_val, y_rt_rain_val
    gc.collect()

    # Cache for next run
    with open(EVALS_CACHE, "w") as f:
        json.dump({"temperature": evals_temp, "rainfall": evals_rain}, f)
    print(f"  Cached training curves -> {EVALS_CACHE}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: GENERATE ALL 8 PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Generating publication-quality plots ...")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1: Training vs Validation Loss (Boosting Rounds)
# ──────────────────────────────────────────────────────────────────────────────
print("  [1/8] Training vs Validation Loss ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Temperature (LightGBM RMSE)
if evals_temp.get("train_rmse"):
    train_loss = evals_temp["train_rmse"]
    val_loss   = evals_temp["valid_rmse"]
    rounds = range(1, len(train_loss) + 1)
    ax1.plot(rounds, train_loss, color=ACCENT_CYAN, linewidth=2, label="Training")
    ax1.plot(rounds, val_loss,   color=ACCENT_ROSE, linewidth=2, label="Validation")
    ax1.fill_between(rounds, train_loss, val_loss, color=ACCENT_ROSE, alpha=0.06)
    gap = abs(val_loss[-1] - train_loss[-1])
    ax1.annotate(f"final gap: {gap:.4f}", xy=(rounds[-1], val_loss[-1]),
                 xytext=(-90, 15), textcoords="offset points",
                 color=ACCENT_AMBER, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=ACCENT_AMBER, lw=0.8))
    _style_axes(ax1, "Temperature (LightGBM) — RMSE", "Boosting Round", "RMSE")
    ax1.legend(frameon=False)

# Rainfall (XGBoost Logloss)
if evals_rain.get("train_logloss"):
    train_loss_r = evals_rain["train_logloss"]
    val_loss_r   = evals_rain["valid_logloss"]
    rounds_r = range(1, len(train_loss_r) + 1)
    ax2.plot(rounds_r, train_loss_r, color=ACCENT_TEAL, linewidth=2, label="Training")
    ax2.plot(rounds_r, val_loss_r,   color=ACCENT_AMBER, linewidth=2, label="Validation")
    ax2.fill_between(rounds_r, train_loss_r, val_loss_r, color=ACCENT_AMBER, alpha=0.06)
    gap_r = abs(val_loss_r[-1] - train_loss_r[-1])
    ax2.annotate(f"final gap: {gap_r:.4f}", xy=(rounds_r[-1], val_loss_r[-1]),
                 xytext=(-90, 15), textcoords="offset points",
                 color=ACCENT_ROSE, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=ACCENT_ROSE, lw=0.8))
    _style_axes(ax2, "Rain Classifier (XGBoost) — Logloss", "Boosting Round", "Logloss")
    ax2.legend(frameon=False)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_training_curve.png")
plt.close()
print(f"    -> {OUT_DIR}/01_training_curve.png")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2: Actual vs Predicted (Temperature)
# ──────────────────────────────────────────────────────────────────────────────
print("  [2/8] Actual vs Predicted (Temperature) ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

for ax, y_true, y_pred, title, color in [
    (ax1, y_max_test, max_preds, "Max Temperature", ACCENT_CYAN),
    (ax2, y_min_test, min_preds, "Min Temperature", ACCENT_TEAL),
]:
    sample_n = min(8000, len(y_true))
    rng_p = np.random.default_rng(42)
    idx = rng_p.choice(len(y_true), sample_n, replace=False)
    y_t = np.array(y_true)[idx]
    y_p = np.array(y_pred)[idx]

    ax.scatter(y_t, y_p, s=8, alpha=0.3, color=color, edgecolors="none")
    lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
    ax.plot(lims, lims, color=ACCENT_ROSE, linewidth=1.5, linestyle="--", label="y = x")

    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    stats = f"R² = {r2:.4f}\nMAE = {mae:.3f}°C\nRMSE = {rmse:.3f}°C"
    ax.text(0.05, 0.95, stats, transform=ax.transAxes, va="top", fontsize=11,
            color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_COLOR,
                      edgecolor=ACCENT_TEAL, alpha=0.85))
    _style_axes(ax, f"Actual vs Predicted — {title}", "Actual (°C)", "Predicted (°C)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_actual_vs_predicted.png")
plt.close()
print(f"    -> {OUT_DIR}/02_actual_vs_predicted.png")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 3: Residual Plot
# ──────────────────────────────────────────────────────────────────────────────
print("  [3/8] Residual Plot ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, y_true, y_pred, title, color in [
    (ax1, y_max_test, max_preds, "Max Temperature", ACCENT_VIOLET),
    (ax2, y_min_test, min_preds, "Min Temperature", ACCENT_CYAN),
]:
    residuals = np.array(y_true) - np.array(y_pred)
    sample_n = min(8000, len(residuals))
    rng_r = np.random.default_rng(42)
    idx = rng_r.choice(len(residuals), sample_n, replace=False)

    ax.scatter(np.array(y_pred)[idx], residuals[idx], s=8, alpha=0.3,
               color=color, edgecolors="none")
    ax.axhline(0, color=ACCENT_ROSE, linewidth=1.5, linestyle="--")

    # Add mean & std annotation
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    ax.text(0.05, 0.95, f"μ = {mu:.3f}°C\nσ = {sigma:.3f}°C",
            transform=ax.transAxes, va="top", fontsize=11, color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL_COLOR,
                      edgecolor=color, alpha=0.85))

    _style_axes(ax, f"Residuals — {title}", "Predicted (°C)", "Residual (Actual − Predicted)")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_residuals.png")
plt.close()
print(f"    -> {OUT_DIR}/03_residuals.png")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 4: Time Series Overlay (Ahmedabad)
# ──────────────────────────────────────────────────────────────────────────────
print("  [4/8] Time Series Overlay (Ahmedabad) ...")

# Locate Ahmedabad grid point in test set
abad_mask = (
    (df.loc[temp_test_idx, "Latitude"]  == 23.5) &
    (df.loc[temp_test_idx, "Longitude"] == 72.5)
)

if abad_mask.sum() > 30:
    # Restore rain_roll7 mean for temp prediction
    df["Rain_roll7_rain3"] = df["Rain_roll7"]
    df["Rain_roll7"] = df["Rain_roll7_mean"]

    abad_test_idx = temp_test_idx[abad_mask.values]
    X_abad = df.loc[abad_test_idx, temp_feature_cols]
    y_abad = df.loc[abad_test_idx, "Max_Temp"].values
    dates_abad = df.loc[abad_test_idx, "Date"].values
    pred_abad = max_temp_model.predict(X_abad, num_iteration=max_temp_model.best_iteration)

    # Sort by date
    sort_idx = np.argsort(dates_abad)
    dates_abad = dates_abad[sort_idx]
    y_abad = y_abad[sort_idx]
    pred_abad = pred_abad[sort_idx]

    # Plot first year
    n_plot = min(365, len(y_abad))

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(dates_abad[:n_plot], y_abad[:n_plot], color=ACCENT_CYAN, linewidth=1.8,
            label="Actual (IMD)", alpha=0.9)
    ax.plot(dates_abad[:n_plot], pred_abad[:n_plot], color=ACCENT_AMBER, linewidth=1.8,
            linestyle="--", label="Predicted (LightGBM)", alpha=0.9)
    _style_axes(ax, "Predicted vs Actual Max Temperature — Ahmedabad (23.5°N, 72.5°E)",
                "Date", "Temperature (°C)")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_timeseries_overlay.png")
    plt.close()
    print(f"    -> {OUT_DIR}/04_timeseries_overlay.png")

    # Restore
    df["Rain_roll7"] = df["Rain_roll7_rain3"]
    df.drop(columns=["Rain_roll7_rain3"], inplace=True)
else:
    print("    (Ahmedabad grid point not found in test set — skipped)")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 5: Confusion Matrix (Rainfall Classifier)
# ──────────────────────────────────────────────────────────────────────────────
print("  [5/8] Confusion Matrix ...")

cm = confusion_matrix(y_rain_test, rain_preds_binary)
class_names = ["No Rain", "Rain"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Raw counts
im1 = ax1.imshow(cm, cmap="mako" if "mako" in plt.colormaps() else "viridis", aspect="auto")
ax1.set_xticks(range(2)); ax1.set_xticklabels(class_names)
ax1.set_yticks(range(2)); ax1.set_yticklabels(class_names)
for i in range(2):
    for j in range(2):
        ax1.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                 color="white", fontsize=14, fontweight="bold")
_style_axes(ax1, f"Confusion Matrix (N = {cm.sum():,})", "Predicted", "Actual")
ax1.grid(False)

# Normalized (%)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
im2 = ax2.imshow(cm_norm, cmap="mako" if "mako" in plt.colormaps() else "viridis",
                 aspect="auto", vmin=0, vmax=100)
ax2.set_xticks(range(2)); ax2.set_xticklabels(class_names)
ax2.set_yticks(range(2)); ax2.set_yticklabels(class_names)
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f"{cm_norm[i, j]:.1f}%", ha="center", va="center",
                 color="white", fontsize=14, fontweight="bold")
_style_axes(ax2, f"Normalized — Acc: {rain_acc*100:.1f}%  F1: {rain_f1:.4f}", "Predicted", "Actual")
ax2.grid(False)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_confusion_matrix.png")
plt.close()
print(f"    -> {OUT_DIR}/05_confusion_matrix.png")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 6: ROC + Precision-Recall Curves
# ──────────────────────────────────────────────────────────────────────────────
print("  [6/8] ROC + Precision-Recall Curves ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_rain_test, rain_probs)
roc_auc = auc(fpr, tpr)
ax1.plot(fpr, tpr, color=ACCENT_CYAN, linewidth=2.5, label=f"AUC = {roc_auc:.3f}")
ax1.plot([0, 1], [0, 1], color=GRID_COLOR, linestyle="--", linewidth=1)
ax1.fill_between(fpr, tpr, alpha=0.08, color=ACCENT_CYAN)
_style_axes(ax1, "ROC Curve — Rainfall Classifier", "False Positive Rate", "True Positive Rate")
ax1.legend(frameon=False, fontsize=12)

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_rain_test, rain_probs)
pr_auc = auc(rec, prec)
ax2.plot(rec, prec, color=ACCENT_TEAL, linewidth=2.5, label=f"PR AUC = {pr_auc:.3f}")
ax2.fill_between(rec, prec, alpha=0.08, color=ACCENT_TEAL)
_style_axes(ax2, "Precision-Recall Curve", "Recall", "Precision")
ax2.legend(frameon=False, fontsize=12)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_roc_pr_curves.png")
plt.close()
print(f"    -> {OUT_DIR}/06_roc_pr_curves.png")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 7: Feature Importance (with Bhadali Vakyo highlighting)
# ──────────────────────────────────────────────────────────────────────────────
print("  [7/8] Feature Importance ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Temperature model importance
temp_imp = pd.DataFrame({
    "Feature": temp_feature_cols,
    "Importance": max_temp_model.feature_importance(importance_type="gain"),
}).sort_values("Importance", ascending=True).tail(20)

temp_colors = [
    ACCENT_AMBER if any(h.lower() in f.lower() for h in ["oni", "dmi", "elevation", "enso", "iod", "coast"])
    else ACCENT_CYAN
    for f in temp_imp["Feature"]
]
ax1.barh(temp_imp["Feature"], temp_imp["Importance"], color=temp_colors)
_style_axes(ax1, "Temperature Model — Feature Importance (Top 20)", "Importance (gain)", "")
ax1.legend(
    handles=[Patch(color=ACCENT_AMBER, label="Climate drivers"), Patch(color=ACCENT_CYAN, label="Base features")],
    frameon=False, loc="lower right",
)

# Rainfall classifier importance
if hasattr(rain_classifier, "feature_importances_"):
    rain_imp_vals = rain_classifier.feature_importances_
else:
    rain_imp_vals = rain_classifier.feature_importance()

rain_imp = pd.DataFrame({
    "Feature": rain_feature_cols,
    "Importance": rain_imp_vals,
}).sort_values("Importance", ascending=True).tail(20)

highlight_set = set(BHADALI_FEATURE_NAMES + NEW_FEATURE_NAMES)
rain_colors = [
    ACCENT_AMBER if f in highlight_set else ACCENT_CYAN
    for f in rain_imp["Feature"]
]
ax2.barh(rain_imp["Feature"], rain_imp["Importance"], color=rain_colors)
_style_axes(ax2, "Rainfall Classifier — Feature Importance (Top 20)", "Importance (gain)", "")
ax2.legend(
    handles=[Patch(color=ACCENT_AMBER, label="Bhadali Vakyo + Advanced"), Patch(color=ACCENT_CYAN, label="Baseline")],
    frameon=False, loc="lower right",
)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_feature_importance.png")
plt.close()
print(f"    -> {OUT_DIR}/07_feature_importance.png")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 8: Spatial Error Map
# ──────────────────────────────────────────────────────────────────────────────
print("  [8/8] Spatial Error Map ...")

# Compute grid-cell-wise MAE for temperature
spatial_df = pd.DataFrame({
    "Latitude":  df.loc[temp_test_idx, "Latitude"].values,
    "Longitude": df.loc[temp_test_idx, "Longitude"].values,
    "Actual":    y_max_test.values,
    "Predicted": max_preds,
})
spatial_df["AbsError"] = np.abs(spatial_df["Actual"] - spatial_df["Predicted"])

grid_errors = spatial_df.groupby(["Latitude", "Longitude"]).agg(
    MAE=("AbsError", "mean"),
    Count=("AbsError", "count"),
).reset_index()
grid_errors = grid_errors[grid_errors["Count"] >= 30]  # require min 30 observations

import plotly.graph_objects as go

fig_sp = go.Figure(go.Scattergeo(
    lat=grid_errors["Latitude"],
    lon=grid_errors["Longitude"],
    mode="markers",
    marker=dict(
        size=7,
        color=grid_errors["MAE"],
        colorscale=[[0, ACCENT_CYAN], [0.5, PANEL_COLOR], [1, ACCENT_ROSE]],
        colorbar=dict(title="MAE (°C)", tickfont=dict(color=TEXT_COLOR)),
        line=dict(width=0),
    ),
    text=[f"({r.Latitude:.1f}°N, {r.Longitude:.1f}°E)<br>MAE: {r.MAE:.2f}°C"
          for _, r in grid_errors.iterrows()],
    hoverinfo="text",
))
fig_sp.update_geos(
    scope="asia", center=dict(lat=22, lon=79), projection_scale=4.5,
    showland=True, landcolor=PANEL_COLOR,
    showocean=True, oceancolor=BG_COLOR,
    showcountries=True, countrycolor=GRID_COLOR,
    bgcolor=BG_COLOR,
)
fig_sp.update_layout(
    title=dict(
        text="Max Temperature Prediction Error — Spatial Distribution (Test Set 2022–2025)",
        font=dict(color=TEXT_COLOR, size=14),
    ),
    paper_bgcolor=BG_COLOR, font=dict(color=TEXT_COLOR),
    margin=dict(l=0, r=0, t=50, b=0),
)

fig_sp.write_html(f"{OUT_DIR}/08_spatial_error_map.html")
print(f"    -> {OUT_DIR}/08_spatial_error_map.html")

# PNG export via kaleido
try:
    fig_sp.write_image(f"{OUT_DIR}/08_spatial_error_map.png", scale=2, width=1000, height=800)
    print(f"    -> {OUT_DIR}/08_spatial_error_map.png")
except Exception as e:
    print(f"    PNG export failed ({e}). Install kaleido: py -m pip install -U kaleido")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: CROSS-CHECK METRICS AGAINST SAVED JSON
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: Metrics Cross-Check ...")
print("=" * 70)

if SAMPLE_FRAC < 1.0:
    print("  ⚠ Skipping cross-check: --sample-frac < 1.0 (metrics will differ from full-data run)")
else:
    TOLERANCE_PCT = 10.0  # allow 10% relative difference

    def _check(label, computed, expected, tol=TOLERANCE_PCT):
        if expected == 0:
            diff_pct = 0 if computed == 0 else 100
        else:
            diff_pct = abs(computed - expected) / abs(expected) * 100
        status = "✓" if diff_pct <= tol else "✗ MISMATCH"
        print(f"  {status}  {label:35s}  computed={computed:.4f}  expected={expected:.4f}  diff={diff_pct:.1f}%")
        return diff_pct <= tol

    all_ok = True

    # Temperature cross-check (against model_metrics.json)
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json", "r") as f:
            saved_temp = json.load(f)
        print("\n  — Temperature (vs model_metrics.json) —")
        all_ok &= _check("Max Temp R²",   max_r2,   saved_temp["max_temp"]["R2"])
        all_ok &= _check("Max Temp MAE",  max_mae,  saved_temp["max_temp"]["MAE"])
        all_ok &= _check("Max Temp RMSE", max_rmse, saved_temp["max_temp"]["RMSE"])
        all_ok &= _check("Min Temp R²",   min_r2,   saved_temp["min_temp"]["R2"])
        all_ok &= _check("Min Temp MAE",  min_mae,  saved_temp["min_temp"]["MAE"])
        all_ok &= _check("Min Temp RMSE", min_rmse, saved_temp["min_temp"]["RMSE"])

    # Rainfall cross-check (against rainfall_metrics.json)
    if os.path.exists("rainfall_metrics.json"):
        with open("rainfall_metrics.json", "r") as f:
            saved_rain = json.load(f)
        print("\n  — Rainfall (vs rainfall_metrics.json) —")
        all_ok &= _check("Classifier Accuracy", rain_acc, saved_rain["classifier"]["accuracy"])
        all_ok &= _check("Classifier F1 Score", rain_f1,  saved_rain["classifier"]["f1_score"])

    # ROC AUC is not in saved metrics — report for reference
    print(f"\n  ℹ  ROC AUC = {roc_auc:.4f}  (not in saved metrics — reference only)")

    if all_ok:
        print("\n  ✓ All cross-checks passed within tolerance.")
    else:
        print(f"\n  ✗ SOME CHECKS FAILED (>{TOLERANCE_PCT}% relative diff).")
        print("    This may indicate a mismatch in the feature-engineering replay.")
        print("    Review lag window calculations and climatology joins.")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
elapsed = time.time() - t_start
print("\n" + "=" * 70)
print("  ALL DONE!")
print("=" * 70)
print(f"""
  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)
  Sample fraction: {SAMPLE_FRAC}

  Generated plots:
    [1] {OUT_DIR}/01_training_curve.png       — Train vs Val loss (real retrain)
    [2] {OUT_DIR}/02_actual_vs_predicted.png  — Scatter: Temp actual vs predicted
    [3] {OUT_DIR}/03_residuals.png            — Residual analysis (bias check)
    [4] {OUT_DIR}/04_timeseries_overlay.png   — Ahmedabad time series
    [5] {OUT_DIR}/05_confusion_matrix.png     — Rainfall classifier confusion
    [6] {OUT_DIR}/06_roc_pr_curves.png        — ROC + Precision-Recall
    [7] {OUT_DIR}/07_feature_importance.png   — Feature importance (Bhadali highlighted)
    [8] {OUT_DIR}/08_spatial_error_map.png    — Spatial MAE distribution

  Temperature Model (LightGBM):
    Max Temp — R² = {max_r2:.4f}  MAE = {max_mae:.3f}°C  RMSE = {max_rmse:.3f}°C
    Min Temp — R² = {min_r2:.4f}  MAE = {min_mae:.3f}°C  RMSE = {min_rmse:.3f}°C

  Rainfall Classifier (XGBoost):
    Accuracy = {rain_acc:.4f}  F1 = {rain_f1:.4f}  ROC AUC = {roc_auc:.4f}
""")
