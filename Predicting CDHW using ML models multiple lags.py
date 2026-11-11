# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 15:04:26 2026

@author: USER
"""

import numpy as np
import pandas as pd
import os
import xarray as xr

input_dir = 'C:/Users/USER/OneDrive - Indian Institute of Science/CDHW Group Project/Datasets/Processed/Final dataset'
CDHW_dir = 'C:/Users/USER/OneDrive - Indian Institute of Science/CDHW Group Project/compound data/upto 2025'

Temp = xr.open_dataset(os.path.join(input_dir, 'rainfall_maxtemperature_monthly_1deg_1951_2025.nc'))
CI = pd.read_csv(os.path.join(input_dir, 'Climate_indices_updated.csv'))
CDHW = xr.open_dataset(os.path.join(CDHW_dir, 'compoundevent_2025_baseline1951_2000.nc'))

# Assume CDHW shape: (time, lat, lon) and CI rows/time match CDHW['time']
CDHW_data = CDHW['compound_event'].astype(int).values
temp_data = Temp['maximum temperature'].values ## should be changed
rain_data = Temp['rainfall'].values ## should be changed

lat_vals = Temp['lat'].values
lon_vals = Temp['lon'].values
n_lat = len(lat_vals)
n_lon = len(lon_vals)

df_lag0 = pd.read_csv(os.path.join(input_dir, 'Lag0_prediction_CDHW.csv'))
df_lag1 = pd.read_csv(os.path.join(input_dir, 'Lag1_prediction_CDHW.csv'))
df_lag2 = pd.read_csv(os.path.join(input_dir, 'Lag2_prediction_CDHW.csv'))
df_lag3 = pd.read_csv(os.path.join(input_dir, 'Lag3_prediction_CDHW.csv'))

dfs = [df_lag0, df_lag1, df_lag2, df_lag3] 

keys = ['time', 'lat', 'lon']

# columns you want ONLY from lag0
base_cols = set(df_lag0.columns)

def keep_only_lagged(df, lag):
    cols_to_keep = keys + [c for c in df.columns if f'_lag{lag}' in c]
    return df[cols_to_keep]

df_lag1_clean = keep_only_lagged(df_lag1, 1)
df_lag2_clean = keep_only_lagged(df_lag2, 2)
df_lag3_clean = keep_only_lagged(df_lag3, 3)

df_all = df_lag0.merge(df_lag1_clean, on=keys, how='inner') \
                .merge(df_lag2_clean, on=keys, how='inner') \
                .merge(df_lag3_clean, on=keys, how='inner')

### Top 5 Features selected from lag models
xgb_features = ['temperature', 'rain', 'lat', 'lon', 'PDO', 'rain_lag1', 'ONI_lag1', 'temperature_lag1', 'PDO_lag1', 'IOD_lag1',
                'ONI_lag2', 'temperature_lag2', 'rain_lag2', 'IOD_lag2', 'IOD_lag3', 'ONI_lag3', 'temperature_lag3', 'NAO_lag3']

# Keep only required columns
required_cols = ['time', 'cdhw'] + xgb_features
df_model = df_all[required_cols].dropna().reset_index(drop=True)

# Sort strictly by time
df_model = df_model.sort_values('time').reset_index(drop=True)

#%% Train and test data preperation
# Split into training and test set with stratified sampling to keep class balance
# Explicit predictor list (example – adjust to your CI names)
y = df_model['cdhw']

X = df_model[xgb_features+ ['time']]

# Define split time index
split_time = 675   # or convert from year if you want

train = df_lag3[df_lag3['time']< split_time]
test = df_lag3[df_lag3['time'] >= split_time]

X_train = X[X['time'] < split_time].drop(columns=['time'], errors = 'ignore')
y_train = y[X['time'] < split_time]

X_test  = X[X['time'] >= split_time].drop(columns=['time'], errors = 'ignore')
y_test  = y[X['time'] >= split_time]

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X = X.drop(columns=['time'], errors = 'ignore')
X_scaled = scaler.transform(X)

#%% Useful Functions

def fit_model_and_predict_proba(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_test  = model.predict_proba(X_test)[:, 1]
    return model, y_proba_train, y_proba_test

from sklearn.metrics import precision_recall_curve
import numpy as np

def optimize_threshold_f1(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1)
    return thresholds[best_idx]

def apply_threshold(y_proba, threshold):
    return (y_proba >= threshold).astype(int)

from sklearn.metrics import brier_score_loss

def probabilistic_scores(y_true, y_proba):
    brier = brier_score_loss(y_true, y_proba)
    clim = brier_score_loss(y_true, np.full_like(y_true, y_true.mean()))
    bss = 1 - (brier / clim)
    return brier, clim, bss

import matplotlib.pyplot as plt

def plot_reliability_train_test(
    y_train, y_proba_train,
    y_test, y_proba_test,
    output_dir,
    label,
    fname="Reliability_Logistic_Train_Test"
):
    import os
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    # Global font settings (journal-ready)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14

    # Compute calibration curves
    prob_true_tr, prob_pred_tr = calibration_curve(
        y_train, y_proba_train, n_bins=15, strategy='quantile'
    )
    prob_true_te, prob_pred_te = calibration_curve(
        y_test, y_proba_test, n_bins=15, strategy='quantile'
    )

    x_max = max(
        prob_true_tr.max(), prob_pred_tr.max(),
        prob_true_te.max(), prob_pred_te.max()
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # ---- Training ----
    axs[0].plot(prob_pred_tr, prob_true_tr, 'o-', label= label)
    axs[0].plot([0, x_max], [0, x_max], '--k', label='Perfect')
    axs[0].set_title('Training set')
    axs[0].set_xlabel('Predicted probability')
    axs[0].set_ylabel('Observed frequency')
    axs[0].grid(True)
    axs[0].legend()

    # ---- Testing ----
    axs[1].plot(prob_pred_te, prob_true_te, 'o-', label= label)
    axs[1].plot([0, x_max], [0, x_max], '--k', label='Perfect')
    axs[1].set_title('Testing set')
    axs[1].set_xlabel('Predicted probability')
    axs[1].grid(True)
    axs[1].legend()

    # Reduce gap between title and panels
    fig.suptitle(
        'Reliability diagram for CDHW prediction',
        fontsize=16, y=0.89, fontweight = 'bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, fname + ".png"),
        dpi=600,
        bbox_inches='tight'
    )
    plt.show()
    plt.close()

from sklearn.metrics import precision_score, recall_score

def spatial_skill_maps(df, lat_vals, lon_vals, y_col, pred_col, min_events=10):
    n_lat, n_lon = len(lat_vals), len(lon_vals)
    prec_grid = np.full((n_lat, n_lon), np.nan)
    recall_grid = np.full((n_lat, n_lon), np.nan)

    for i, lat in enumerate(lat_vals):
        for j, lon in enumerate(lon_vals):
            subset = df[(df['lat'] == lat) & (df['lon'] == lon)]
            if subset[y_col].sum() < min_events:
                continue
            prec_grid[i, j] = precision_score(subset[y_col], subset[pred_col])
            recall_grid[i, j] = recall_score(subset[y_col], subset[pred_col])

    return prec_grid, recall_grid

# =============================================================================
# FORMAT AXES AS DEGREE LABELS
# =============================================================================

def format_geo_axes(ax):
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    # Format longitude (°E)
    ax.set_xticklabels([f"{abs(x):.0f}°E" for x in xticks])

    # Format latitude (°N)
    ax.set_yticklabels([f"{abs(y):.0f}°N" for y in yticks])

    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')


# Plot spatial recall map
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

#%% XGBoost
output_dir = 'C:/Users/USER/OneDrive - Indian Institute of Science/CDHW Group Project/ML Results updated/XGBoost Multilag'
# Define parameter space
param_dist = {
    'n_estimators': [400, 700],
    'max_depth': [4, 5],
    'learning_rate': [0.02, 0.05],
    'min_child_weight': [3, 5, 8],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'gamma': [0, 1, 2],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 3, 5]
}

from sklearn.metrics import fbeta_score, make_scorer
fbeta = make_scorer(fbeta_score, beta=2)

# Base XGBoost model (imbalance handled correctly)
import xgboost as xgb

n_cores = os.cpu_count()
n_jobs_safe = max(1, n_cores - 2)   # e.g. if you have 8 cores, uses 6
print(f"Using {n_jobs_safe} of {n_cores} cores")
base_model = xgb.XGBClassifier( 
    objective='binary:logistic',
    eval_metric='aucpr',
    scale_pos_weight=1.5*(y_train == 0).sum() / (y_train == 1).sum(),
    max_delta_step=1,   # 🔥 stabilizes rare-event probabilities
    random_state=42,
    n_jobs=n_jobs_safe
)


# Hyperparameter search (TRAIN ONLY)
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=40,                 # enough for journals
    scoring='neg_brier_score',  # PR-AUC
    cv=5,                      # ideally TimeSeriesSplit
    verbose=2,
    random_state=42,
    n_jobs=n_jobs_safe
)

search.fit(X_train_scaled, y_train)

#Best model & parameters
best_xgb = search.best_estimator_

from sklearn.calibration import CalibratedClassifierCV

cal_xgb = CalibratedClassifierCV(
    best_xgb,
    method='isotonic',   
    cv=5
)

final_xgb, y_proba_train, y_proba_test = fit_model_and_predict_proba(cal_xgb, X_train_scaled, y_train, X_test_scaled)

best_threshold = optimize_threshold_f1(y_train, y_proba_train)
event_rate = y_train.mean()
threshold_freq = np.quantile(y_proba_train, 1 - event_rate)

print("F1 threshold:", best_threshold)
print("Frequency-consistent threshold:", threshold_freq)

best_threshold = optimize_threshold_f1(y_train, y_proba_train)
print("Optimal probability threshold:", best_threshold)

import joblib
import json

# ── SAVE BEST XGBoost MODEL ───────────────────────────────────────────────────

# 1. Save final calibrated model (CalibratedClassifierCV wrapping XGBoost)
model_path = os.path.join(output_dir, 'final_xgb_calibrated_multilag.pkl')
joblib.dump(final_xgb, model_path)
print(f"Calibrated model saved: {model_path}")

# 2. Save the underlying best XGBoost (before calibration) -- native format
xgb_path = os.path.join(output_dir, 'best_xgb_uncalibrated_multilag.json')
best_xgb.save_model(xgb_path)
print(f"XGBoost native model saved: {xgb_path}")

# 3. Save best hyperparameters for reference
params_path = os.path.join(output_dir, 'best_xgb_params.json')
with open(params_path, 'w') as f:
    json.dump(search.best_params_, f, indent=4)
print(f"Best params saved: {params_path}")

# 4. Save the scaler
scaler_path = os.path.join(output_dir, 'scaler_multilag.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved: {scaler_path}")

# 5. Save feature column names
feature_path = os.path.join(output_dir, 'feature_columns_multilag.json')
with open(feature_path, 'w') as f:
    json.dump(list(X_train.columns), f, indent=4)
print(f"Feature columns saved: {feature_path}")

# 6. Save threshold values
threshold_path = os.path.join(output_dir, 'thresholds_multilag.json')
thresholds = {
    'threshold_freq': float(threshold_freq),
    'best_threshold': float(best_threshold)
}
with open(threshold_path, 'w') as f:
    json.dump(thresholds, f, indent=4)
print(f"Thresholds saved: {threshold_path}")

# 7. Print summary
print(f"\nBest CV score (neg Brier): {search.best_score_:.4f}")
print(f"Best parameters:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

# Apply threshold
y_pred_train = apply_threshold(y_proba_train , threshold_freq)
y_pred_test  = apply_threshold(y_proba_test , threshold_freq)

# Classification report (journal-ready)
from sklearn.metrics import classification_report
print("TRAIN PERFORMANCE")
print(classification_report(y_train, y_pred_train, digits=3))
print("TEST PERFORMANCE")
print(classification_report(y_test, y_pred_test, digits=3))

# Brier score (mandatory for probabilistic models)
brier_train, brier_clim_train, bss_train = probabilistic_scores(y_train, y_proba_train)
brier_test, brier_clim_test, bss_test = probabilistic_scores(y_test, y_proba_test)

print(f"Brier score (train): {brier_train:.4f}")
print(f"Brier score (test): {brier_test:.4f}")
print("Train Climatological Brier:", brier_clim_train)
print("Test Climatological Brier:", brier_clim_test)
print(f"Brier Skill Score Train: {bss_train:.3f}")
print(f"Brier Skill Score Test: {bss_test:.3f}")

plot_reliability_train_test(
    y_train, y_proba_train,
    y_test, y_proba_test,
    output_dir,
    'XGBoost',
    fname="Reliability_XGBoost_Train_Test"
)

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# ---- TRAIN ----
precision_tr, recall_tr, _ = precision_recall_curve(y_train, y_proba_train)
ap_tr = average_precision_score(y_train, y_proba_train)

axes[0].plot(recall_tr, precision_tr, color='tab:blue',
             label=f'Train (AUPRC = {ap_tr:.3f})')
axes[0].set_title('Training set')
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].grid(True)
axes[0].legend()

# ---- TEST ----
precision_te, recall_te, _ = precision_recall_curve(y_test, y_proba_test)
ap_te = average_precision_score(y_test, y_proba_test)

axes[1].plot(recall_te, precision_te, color='tab:red',
             label=f'Test (AUPRC = {ap_te:.3f})')
axes[1].set_title('Testing set')
axes[1].set_xlabel('Recall')
axes[1].grid(True)
axes[1].legend()

plt.suptitle('Precision–Recall Curves for CDHW Prediction', fontsize=14, y = 0.93)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'PRCurves_XGB_CDHW.png'), dpi = 600,  bbox_inches='tight')
plt.show()

from sklearn.metrics import precision_score, recall_score
import numpy as np

metrics = {
    'Precision': [
        precision_score(y_train, y_pred_train),
        precision_score(y_test, y_pred_test)
    ],
    'Recall': [
        recall_score(y_train, y_pred_train),
        recall_score(y_test, y_pred_test)
    ]
}


metrics_df = pd.DataFrame(metrics)

# Save
metrics_df.to_csv(
    os.path.join(output_dir, 'Precision_Recall_XGB_CDHW.csv'),
    index=False
)

labels = ['Train', 'Test']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(6,5))
ax.bar(x - width/2, metrics['Precision'], width, label='Precision')
ax.bar(x + width/2, metrics['Recall'], width, label='Recall')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title('Precision and Recall at Optimal Threshold')
ax.legend()
ax.grid(axis='y')
plt.savefig(os.path.join(output_dir, 'PRvalues_XGB_CDHW.png'), dpi = 600,  bbox_inches='tight')
plt.show()


# Add predictions back
train = train.copy()
train['y_pred'] = y_pred_train
test = test.copy()
test['y_pred'] = y_pred_test

prec_grid_train, recall_grid_train = spatial_skill_maps(train, Temp['lat'].values, Temp['lon'].values, 'cdhw', 'y_pred', min_events=10)
prec_grid_test, recall_grid_test = spatial_skill_maps(test, Temp['lat'].values, Temp['lon'].values, 'cdhw', 'y_pred', min_events=10)

import matplotlib.pyplot as plt
import geopandas as gpd

# Load shapefile
shp_path = r"D:\PhD Course Work\WR 201 Watershed Modeling\Project\DATA\Administrative_boundary\STATE_BOUNDARY.shp"
shp_gdf = gpd.read_file(shp_path)

if shp_gdf.crs != "EPSG:4326":
    shp_gdf = shp_gdf.to_crs("EPSG:4326")

extent = [lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()]

fig, axes = plt.subplots(
    2, 2,
    figsize=(10,10),
    sharex=True,
    sharey=True,
    constrained_layout=False
)

plots = [
    (recall_grid_train, 'Recall (Training)', axes[0, 0]),
    (prec_grid_train,   'Precision (Training)', axes[0, 1]),
    (recall_grid_test,  'Recall (Testing)', axes[1, 0]),
    (prec_grid_test,    'Precision (Testing)', axes[1, 1])
]

for data, title, ax in plots:
    im = ax.imshow(
        data,
        origin='lower',
        extent=extent,
        cmap='Blues',
        aspect='auto',
        vmin=0,
        vmax=1
    )
    shp_gdf.boundary.plot(ax=ax, linewidth=1, edgecolor='k')
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

# 🔹 Create space on the right for colorbar
fig.subplots_adjust(right=0.88, wspace=0.05, hspace=0.19)

# 🔹 Dedicated colorbar axis (outside plots)
cax = fig.add_axes([0.90, 0.20, 0.02, 0.60])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Skill Score')

plt.suptitle(
    'Spatial Skill of XGBoost for CDHW Prediction',
    fontsize=16, y=0.94 , fontweight = 'bold'
)
plt.savefig(os.path.join(output_dir, 'PRSpatialMaps_XGB_CDHW.png'), dpi = 600,  bbox_inches='tight')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Built-in importance (gain is preferred for papers)
importance_gain = best_xgb.get_booster().get_score(importance_type='gain')

fi_gain = pd.DataFrame({
    'Feature':  X_train.columns,                              # your actual column names
    'Gain':     best_xgb.feature_importances_                 # uses gain by default
}).sort_values('Gain', ascending=False)
print(fi_gain.head(10))

print(fi_gain.head(10))

plt.figure(figsize=(7,5))
plt.barh(fi_gain['Feature'][:10][::-1], fi_gain['Gain'][:10][::-1])
plt.xlabel('Gain')
plt.title('XGBoost Feature Importance (Gain)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Featureimp_XGB_CDHW.png'), dpi = 600,  bbox_inches='tight')
plt.show()


# Predict probabilities for FULL dataset
y_proba_full = final_xgb.predict_proba(X_scaled)[:, 1]

# Apply optimal threshold (from training)
y_pred_full = (y_proba_full >= threshold_freq).astype(int)

df_model = df_model.copy()
df_model['y_pred_xgb'] = y_pred_full

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os

cm = confusion_matrix(df_model['cdhw'], df_model['y_pred_xgb'])

plt.rcParams['font.family'] = 'Times New Roman'

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm, cmap='Blues')

# Threshold to switch text color
threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > threshold else "black"
        ax.text(
            j, i, f"{cm[i, j]:d}",
            ha="center", va="center",
            fontsize=18,
            color=color
        )

# Axis labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['No CDHW', 'CDHW'], fontsize=12)
ax.set_yticklabels(['No CDHW', 'CDHW'], fontsize=12)

ax.set_xlabel('Predicted', fontsize=13)
ax.set_ylabel('Observed', fontsize=13)

ax.set_title(
    'Confusion Matrix (XGBoost – Full Dataset)',
    fontsize=14,
    pad=10,
    fontweight='bold'
)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Number of samples', fontsize=12)
cbar.ax.tick_params(labelsize=11)

plt.tight_layout(rect=[0, 0, 0.92, 0.95])

plt.savefig(
    os.path.join(output_dir, 'ConfusionMatrix_XGB_Full.png'),
    dpi=600,
    bbox_inches='tight'
)
plt.show()


import numpy as np
from sklearn.metrics import precision_score, recall_score

lat_vals = Temp['lat'].values
lon_vals = Temp['lon'].values

n_lat, n_lon = len(lat_vals), len(lon_vals)

# df_lag1 must already contain:
# columns: ['lat', 'lon', 'cdhw', 'y_pred']
# best: no merge needed

cdhw_grid = np.full((n_lat, n_lon), np.nan)
cdhw_pred_grid = np.full((n_lat, n_lon), np.nan)
prec_grid = np.full((n_lat, n_lon), np.nan)
recall_grid = np.full((n_lat, n_lon), np.nan)

for i, lat in enumerate(lat_vals):
    for j, lon in enumerate(lon_vals):
        subset = df_model[
            (df_model['lat'] == lat) &
            (df_model['lon'] == lon)
        ]

        # Skip empty grids
        if subset.empty:
            continue

        # Observed & predicted CDHW counts
        cdhw_grid[i, j] = subset['cdhw'].sum()
        cdhw_pred_grid[i, j] = subset['y_pred_xgb'].sum()

        # Skill only if both classes exist
        if subset['cdhw'].nunique() > 1:
            prec_grid[i, j] = precision_score(
                subset['cdhw'], subset['y_pred_xgb'], zero_division=0
            )
            recall_grid[i, j] = recall_score(
                subset['cdhw'], subset['y_pred_xgb'], zero_division=0
            )


import geopandas as gpd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

shp_path = r"D:\PhD Course Work\WR 201 Watershed Modeling\Project\DATA\Administrative_boundary\STATE_BOUNDARY.shp"
shp_gdf = gpd.read_file(shp_path)

if shp_gdf.crs != "EPSG:4326":
    shp_gdf = shp_gdf.to_crs("EPSG:4326")

extent = [lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()]

import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

cdhw_colors = ['blue', 'cyan', 'yellow', 'orange', 'red']
cdhw_cmap = LinearSegmentedColormap.from_list(
    'CDHW_cmap', cdhw_colors, N=256
)

# Define bins based on event counts (adjust if needed)
ul1 = np.nanmax(cdhw_pred_grid)
ul2 = np.nanmax(cdhw_grid)
ul = max(ul1,ul2,120)
cdhw_bounds = np.unique(
    np.concatenate((
        [0, 1],
        np.arange(5, ul+5, 5)
    ))
)

cdhw_norm = mcolors.BoundaryNorm(cdhw_bounds, cdhw_cmap.N)

skill_cmap = 'Blues'
skill_norm = mcolors.Normalize(vmin=0, vmax=1)

plt.rcParams['font.family'] = 'Times New Roman'

from shapely.vectorized import contains

lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)

mask = contains(
    shp_gdf.unary_union,
    lon2d,
    lat2d
)

cdhw_grid_masked = np.where(mask, cdhw_grid, np.nan)
cdhw_pred_grid_masked = np.where(mask, cdhw_pred_grid, np.nan)
prec_grid_masked = np.where(mask, prec_grid, np.nan)
recall_grid_masked = np.where(mask, recall_grid, np.nan)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.subplots_adjust(
    wspace=0.05,
    hspace=0.25,
    right=0.92
)

extent = [lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()]

# --- Observed CDHW ---
im1 = axes[0, 0].imshow(
    cdhw_grid_masked,
    origin='lower',
    extent=extent,
    cmap=cdhw_cmap,
    norm=cdhw_norm
)
axes[0, 0].set_title('(a) Observed CDHW Events')

cb1 = fig.colorbar(im1, ax=axes[0, 0], shrink=0.85)
#cb1.set_label('CDHW Event Count')

# --- Predicted CDHW ---
im2 = axes[0, 1].imshow(
    cdhw_pred_grid_masked,
    origin='lower',
    extent=extent,
    cmap=cdhw_cmap,
    norm=cdhw_norm
)
axes[0, 1].set_title('(b) Predicted CDHW Events')

cb2 = fig.colorbar(im2, ax=axes[0, 1], shrink=0.85)
#cb2.set_label('CDHW Event Count')

# --- Precision ---
im3 = axes[1, 0].imshow(
    prec_grid_masked,
    origin='lower',
    extent=extent,
    cmap=skill_cmap,
    norm=skill_norm
)
axes[1, 0].set_title('(d) Precision')

cb3 = fig.colorbar(im3, ax=axes[1, 0], shrink=0.85)
#cb3.set_label('Precision')

# --- Recall ---
im4 = axes[1, 1].imshow(
    recall_grid_masked,
    origin='lower',
    extent=extent,
    cmap=skill_cmap,
    norm=skill_norm
)
axes[1, 1].set_title('(e) Recall')

cb4 = fig.colorbar(im4, ax=axes[1, 1], shrink=0.85)
#cb4.set_label('Recall')

# --- Overlay boundaries ---
for ax in axes.flat:
    shp_gdf.boundary.plot(ax=ax, linewidth=0.8, edgecolor='k')
    format_geo_axes(ax)
    
plt.savefig(
    os.path.join(output_dir, 'CDHW_SpatialMaps_Final1.png'),
    dpi=600,
    bbox_inches='tight'
)
plt.show()

prec_grid = np.where(
    (prec_grid >= 0.5),
    1, np.nan
)

num_prec_grids = np.nansum(prec_grid)

rec_grid = np.where(
    (recall_grid >= 0.5),
    1, np.nan
)

num_rec_grids = np.nansum(rec_grid)
print(f"Number of grid points with Precision > 0.5: {int(num_prec_grids)}")
print(f"Number of grid points with Recall > 0.5: {int(num_rec_grids)}")

metrics_full = {
    'Precision': [
        precision_score(df_model['cdhw'], df_model['y_pred_xgb'])
    ],
    'Recall': [
        recall_score(df_model['cdhw'], df_model['y_pred_xgb'])
    ]
}

print(metrics_full)

brier, brier_clim, bss = probabilistic_scores(df_model['cdhw'], y_proba_full)

print(f"Brier score : {brier:.4f}")
print("Climatological Brier:", brier_clim)
print(f"Brier Skill Score: {bss:.3f}")

df_model.to_csv(os.path.join(input_dir, 'XGB_multilag_prediction_CDHW.csv'))

#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os

cm_raw = confusion_matrix(df_model['cdhw'], df_model['y_pred_xgb'])

# ── Compute percentages ───────────────────────────────────────────────────────
cm_pct  = cm_raw / cm_raw.sum() * 100          # % of total samples
cm_row  = cm_raw / cm_raw.sum(axis=1, keepdims=True) * 100  # % per true class (row-normalised)

plt.rcParams['font.family'] = 'Times New Roman'

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

titles = [
    '% of Total Samples',
    '(f) Confusion Matrix',
]
cms = [cm_pct, cm_row]

for ax, data, title in zip(axes, cms, titles):

    im = ax.imshow(data, cmap='Blues', vmin=0, vmax=100)

    threshold = 50   # switch text colour at 50%

    for i in range(2):
        for j in range(2):
            raw  = cm_raw[i, j]
            pct  = data[i, j]
            color = 'white' if pct > threshold else 'black'

            # Main percentage value
            ax.text(j, i ,
                    f'{pct:.1f}%',
                    ha='center', va='center',
                    fontsize=20, fontweight='bold',
                    color=color)

            # Raw count below in smaller font
            ax.text(j, i + 0.22,
                    f'({raw:,})',
                    ha='center', va='center',
                    fontsize=16,
                    color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No CDHW', 'CDHW'], fontsize=13)
    ax.set_yticklabels(['No CDHW', 'CDHW'], fontsize=13, rotation = 90)
    ax.set_xlabel('Predicted', fontsize=18, fontweight = 'bold')
    ax.set_ylabel('Observed', fontsize=18, fontweight = 'bold', rotation = 90)
    ax.set_title(title, fontsize=20, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Percentage (%)', fontsize=16)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_ticks([0, 25, 50, 75, 100])

    # Add box labels in corners
    #labels = [['TN', 'FP'], ['FN', 'TP']]
    # for i in range(2):
    #     for j in range(2):
    #         ax.text(j + 0.42, i - 0.42,
    #                 labels[i][j],
    #                 ha='right', va='top',
    #                 fontsize=10, color='grey',
    #                 fontstyle='italic')

fig.suptitle('Confusion Matrix — XGBoost Multilag',
             fontsize=14, fontweight='bold', y=0.96)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ConfusionMatrix_XGB_Pct.png'),
            dpi=600, bbox_inches='tight')
plt.show()

# ── Print summary ─────────────────────────────────────────────────────────────
tn, fp, fn, tp = cm_raw.ravel()
print(f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
print(f"Total correct: {(tn+tp)/cm_raw.sum()*100:.1f}%")
print(f"CDHW recall (hit rate):    {tp/(tp+fn)*100:.1f}%")
print(f"CDHW precision:            {tp/(tp+fp)*100:.1f}%")


#%% SHAP -- NEW SECTION: add after all three models are trained

import shap, os
import matplotlib.pyplot as plt

shap_output_dir = 'C:/Users/USER/OneDrive - Indian Institute of Science/CDHW Group Project/ML Results updated/XGBoost Multilag'
os.makedirs(shap_output_dir, exist_ok=True)

feature_names = list(X_train.columns)

# Subsample test set for speed -- 2000 samples gives stable SHAP values
np.random.seed(42)
test_df     = pd.DataFrame(X_test_scaled, columns=feature_names)
sample_idx  = np.random.choice(len(test_df), size=min(2000, len(test_df)), replace=False)
test_sample = test_df.iloc[sample_idx]

# --- 3. XGBOOST ---
xgb_base        = final_xgb.calibrated_classifiers_[0].estimator
explainer_xgb   = shap.TreeExplainer(xgb_base)
shap_values_xgb = explainer_xgb(test_sample)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb, test_sample, feature_names=feature_names, plot_type='bar', show=False)
plt.title('SHAP Feature Importance - XGBoost', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(shap_output_dir, 'SHAP_XGB_bar.png'), dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 6))
shap.summary_plot(shap_values_xgb, test_sample, feature_names=feature_names, show=False)
plt.title('(c) SHAP Beeswarm', fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(shap_output_dir, 'SHAP_XGB_beeswarm.png'), dpi=600, bbox_inches='tight')
plt.show()

# --- 4. CROSS-MODEL COMPARISON ---
mean_shap_xgb = np.abs(shap_values_xgb.values).mean(axis=0)

shap_compare = pd.DataFrame({
    'Feature':  feature_names,
    'XGB_SHAP': mean_shap_xgb
}).sort_values('XGB_SHAP', ascending=False)

print(shap_compare.to_string(index=False))
shap_compare.to_csv(os.path.join(shap_output_dir, 'SHAP_xgb.csv'), index=False)

#%%
plt.figure(figsize=(6, 6))

shap.summary_plot(
    shap_values_xgb,
    test_sample,
    feature_names=feature_names,
    max_display=14,
    show=False
)

# ✅ Modify SHAP colorbar
fig = plt.gcf()
for ax in fig.axes:
    if ax.get_ylabel() == "Feature value":
        ax.tick_params(labelsize=16)
        ax.set_ylabel("Feature value", fontsize=18)

# ✅ Axis labels
plt.xlabel('SHAP values', fontsize=20, fontweight = 'bold')
plt.ylabel('Features', fontsize=20, fontweight = 'bold')

# ✅ Tick fonts
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# ✅ Title
plt.title('(c) SHAP Beeswarm', fontsize=28)

plt.tight_layout()

plt.savefig(
    os.path.join(shap_output_dir, 'SHAP_XGB_beeswarm1.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.show()































