# -*- coding: utf-8 -*-
"""
STEP C - XGBoost CDHW forecast models: lead 0, lead 1, lead 2, lead 3 and the multi-lead model, blocked out-of-sample validation.
Spyder cell-based script. NOT RUN by the author of this file.

Reads the fold-wise bias-corrected fields saved by A_save_bias_corrected.py (QDM without caps, fitted without the held-out block).
Every lead-L file is treated as the forecast of the single valid month = initialization month + L.

Models (predictors as in the original scripts, plus the two base rates of the recent method)
  lead L (L = 0, 1, 2, 3)  as CDHW_fcst_lead_final.py: temperature and rain of the lead-L forecast of the target month, lat, lon, and
                           the climate indices PDO, ONI, IOD, NAO of the TARGET month (as in that script)
  multilead                as Forecast XGB CFSv2 IMD Verification Final.py (Eq. 3, 18 predictors): forecasts of one initialization M for
                           target M+3: lag 0 <- lead 3, lag 1 <- lead 2, lag 2 <- lead 1, lag 3 <- lead 0; climate indices of the
                           INITIALIZATION month M (persistence)
  all models               + base_rate (cell) and base_rate_month (cell x target calendar month): observed CDHW frequency 1951-1981
Method (as rolling_origin_pipeline.py, 4 classes): 5 blocks of target years held out in turn (3-month buffer); XGBoost search on the
training rows; out-of-fold probabilities (5 time blocks, buffer); class-wise sigmoid calibration and class-wise threshold, the classes
defined by the 1951-1981 cell-month frequency. The RAW (uncalibrated) probability of the held-out rows is saved as well.

Output: Final_Methodology\03_xgboost_lead0 ... 03_xgboost_lead3, 03_xgboost_multilead, each with
        rolling_origin_predictions.csv, rolling_origin_fold_info.csv, rolling_origin_scores.csv; and in Final_Methodology:
        C_all_configs_summary.csv, C_runtime.csv
"""

#%% Cell 1: imports, paths, settings
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score, brier_score_loss, average_precision_score
import xgboost as xgb

warnings.filterwarnings("ignore")
T0 = time.time()
runtime = []


def tick(label, t_start):
    """Print and store the time taken by one step (minutes) and the running total."""
    mins = (time.time() - t_start) / 60
    runtime.append({"step": label, "minutes": round(mins, 2)})
    print("[time] %-46s %6.1f min | total so far %6.1f min" % (label, mins, (time.time() - T0) / 60))


ONEDRIVE = r"C:\Users\USER\OneDrive - Indian Institute of Science\CDHW Group Project"
CDHW_NC = os.path.join(ONEDRIVE, r"compound data\upto 2025\compoundevent_2025_baseline1951_2000.nc")
CI_CSV = os.path.join(ONEDRIVE, r"Datasets\Processed\Final dataset\Climate_indices_updated.csv")
SHP = (r"D:\PhD Course Work\WR 201 Watershed Modeling\Project\DATA"
       r"\Administrative_boundary\STATE_BOUNDARY.shp")
ROOT = r"E:\CDHW\BAMS Revision\Revision 1\Claude\Final_Methodology"
BC_DIR = os.path.join(ROOT, "01_bias_corrected_data")

# configuration: (name, kind, K = months between initialization and target, results folder)
CONFIGS = [("lead0", "lead", 0, "03_xgboost_lead0"), ("lead1", "lead", 1, "03_xgboost_lead1"),
           ("lead2", "lead", 2, "03_xgboost_lead2"), ("lead3", "lead", 3, "03_xgboost_lead3"),
           ("multilead", "multi", 3, "03_xgboost_multilead")]
OUT_DIRS = {c[0]: os.path.join(ROOT, c[3]) for c in CONFIGS}
for _d in OUT_DIRS.values():
    os.makedirs(_d, exist_ok=True)

BLOCKS = [(1982, 1990), (1991, 1999), (2000, 2008), (2009, 2017), (2018, 2025)]   # held-out TARGET years
BUFFER_MONTHS = 3
N_JOBS = max(1, (os.cpu_count() or 2) - 2)
CLASS_EDGES = [0.0, 0.05, 0.10]       # 4 classes of the 1951-81 cell-month frequency: 0 | (0, 0.05] | (0.05, 0.10] | > 0.10
MIN_EVENTS_CLASS = 100                # a class with fewer training events than this uses the global calibration and threshold
PARAM_DIST = {
    "n_estimators": [400, 700], "max_depth": [4, 5], "learning_rate": [0.02, 0.05],
    "min_child_weight": [3, 5, 8], "subsample": [0.7, 0.8], "colsample_bytree": [0.7, 0.8],
    "gamma": [0, 1, 2], "reg_alpha": [0, 0.1, 1], "reg_lambda": [1, 3, 5],
    "max_delta_step": [1, 2, 3, 5],
}
print("Folds:", BLOCKS, "| buffer:", BUFFER_MONTHS, "months | cores:", N_JOBS)

#%% Cell 2: grid, land mask, initialization months, observed CDHW, climate indices
t_cell = time.time()
first = xr.open_dataset(os.path.join(BC_DIR, "BC_fold1_rain_lead0.nc"))
lat, lon = first.lat.values, first.lon.values
nlat, nlon = len(lat), len(lon)
init_times = pd.DatetimeIndex(first.time.values)                      # initialization months (the same for every file)

shp = gpd.read_file(SHP).to_crs("EPSG:4326")
poly = shp.dissolve().geometry.iloc[0]
land = np.array([[poly.contains(Point(lo, la)) for lo in lon] for la in lat])
print("Land cells:", int(land.sum()), "| initialization months:", len(init_times))


def month_start(da):
    t = pd.DatetimeIndex(da.time.values).to_period("M").to_timestamp()
    return da.assign_coords(time=t)


cd = month_start(xr.open_dataset(CDHW_NC)["compound_event"].reset_coords(drop=True))
cd = cd.reindex(lat=lat, lon=lon, method="nearest", tolerance=1e-3).transpose("time", "lat", "lon")

CI = pd.read_csv(CI_CSV)
CI["m"] = pd.to_datetime(CI["Month"], format="%b").dt.month
CI = CI.set_index(["Year", "m"])


def ci_series(ts):
    """Climate indices (PDO, ONI, IOD, NAO) of the months in ts, as arrays of the length of ts (NaN if missing)."""
    out = {n: np.full(len(ts), np.nan) for n in ("PDO", "ONI", "IOD", "NAO")}
    for i, t in enumerate(ts):
        if (t.year, t.month) in CI.index:
            for n in out:
                out[n][i] = CI.loc[(t.year, t.month), n]
    return out


# ---- observed CDHW climatology 1951-1981 (base rates, classes and BSS reference); no year of 1982-2025 enters it ----
cd_vals = cd.values
cd_t = pd.DatetimeIndex(cd.time.values)
pre = (cd_t.year >= 1951) & (cd_t.year <= 1981)
clim_cm = np.full((12, nlat, nlon), np.nan)
for m in range(1, 13):
    clim_cm[m - 1] = np.nanmean(cd_vals[pre & (cd_t.month == m)], axis=0)
clim_cell = np.nanmean(cd_vals[pre], axis=0)
season_idx = np.array([2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])                  # month 1..12 -> 0 FMAM, 1 JJAS, 2 ONDJ (as in Fig. 5)
clim_season = np.full((3, nlat, nlon), np.nan)
for s in range(3):
    months_s = [m for m in range(1, 13) if season_idx[m - 1] == s]
    clim_season[s] = np.nanmean(cd_vals[pre & np.isin(cd_t.month, months_s)], axis=0)
tick("Cell 2 data", t_cell)

#%% Cell 3: rows (one per initialization month and land cell) and predictors of every configuration
def make_rows(K):
    """Rows of a configuration: forecast initialized in month M, target month M + K; the label is the observed CDHW there."""
    target_times = init_times + pd.DateOffset(months=K)
    truth_all = cd.reindex(time=target_times).values
    r_init, r_i, r_j = np.where(land[None, :, :] & np.isfinite(truth_all))       # ordered by time, then cell
    return {"K": K, "target_times": target_times, "r_init": r_init, "r_i": r_i, "r_j": r_j,
            "y": truth_all[r_init, r_i, r_j].astype(int),
            "r_target": target_times[r_init].values, "r_tyear": target_times.year.values[r_init]}


def build_X(kind, K, f, rw, ci_target, ci_init):
    """Predictors of a configuration (f: corrected fields {('rain', lead): array, ('tmax', lead): array})."""
    ri = rw["r_init"]
    R = lambda L: f[("rain", L)][ri, rw["r_i"], rw["r_j"]]
    T = lambda L: f[("tmax", L)][ri, rw["r_i"], rw["r_j"]]
    if kind == "lead":                         # CDHW_fcst_lead_final.py: lead-K forecast of the target month, indices of the target month
        cols = {"temperature": T(K), "rain": R(K), "lat": lat[rw["r_i"]], "lon": lon[rw["r_j"]],
                "PDO": ci_target["PDO"][ri], "ONI": ci_target["ONI"][ri], "IOD": ci_target["IOD"][ri], "NAO": ci_target["NAO"][ri]}
    else:                                      # Eq. 3 (18 predictors): lags 0..3 <- leads 3..0 of one initialization, indices at the initialization
        c = {n: ci_init[n][ri] for n in ci_init}
        cols = {"temperature": T(3), "rain": R(3), "lat": lat[rw["r_i"]], "lon": lon[rw["r_j"]], "PDO": c["PDO"],
                "rain_lag1": R(2), "ONI_lag1": c["ONI"], "temperature_lag1": T(2), "PDO_lag1": c["PDO"], "IOD_lag1": c["IOD"],
                "ONI_lag2": c["ONI"], "temperature_lag2": T(1), "rain_lag2": R(1), "IOD_lag2": c["IOD"],
                "IOD_lag3": c["IOD"], "ONI_lag3": c["ONI"], "temperature_lag3": T(0), "NAO_lag3": c["NAO"]}
    return pd.DataFrame(cols)


rows_by = {c[0]: make_rows(c[2]) for c in CONFIGS}
ci_init = ci_series(init_times)                                                  # indices of the initialization month
ci_target = {c[0]: ci_series(init_times + pd.DateOffset(months=c[2])) for c in CONFIGS}   # indices of the target month
for c in CONFIGS:
    print("%-10s: %d rows | event rate %.3f" % (c[0], len(rows_by[c[0]]["y"]), rows_by[c[0]]["y"].mean()))

#%% Cell 4: the blocked loop (for every fold and configuration: training, calibration, threshold, prediction)
preds = {c[0]: [] for c in CONFIGS}
fold_info = {c[0]: [] for c in CONFIGS}
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

for k, (y0, y1) in enumerate(BLOCKS, start=1):
    t_fold = time.time()
    print("\n===== Fold %d: held-out target years %d-%d =====" % (k, y0, y1))
    ex_start = pd.Timestamp(y0, 1, 1) - pd.DateOffset(months=BUFFER_MONTHS)    # held-out block plus buffer
    ex_end = pd.Timestamp(y1, 12, 1) + pd.DateOffset(months=BUFFER_MONTHS)

    fields = {}                                                                # saved corrected fields of this fold
    for var in ("rain", "tmax"):
        for L in range(4):
            fields[(var, L)] = xr.open_dataset(os.path.join(BC_DIR, "BC_fold%d_%s_lead%d.nc" % (k, var, L)))["corrected"].values

    for name, kind, K, _ in CONFIGS:
        rw = rows_by[name]
        y_all, r_target, r_tyear = rw["y"], rw["r_target"], rw["r_tyear"]
        print("  --- %s ---" % name)

        # 1) predictors and the split of the rows
        X = build_X(kind, K, fields, rw, ci_target[name], ci_init)
        finite = np.isfinite(X.values).all(axis=1)
        test = (r_tyear >= y0) & (r_tyear <= y1) & finite                          # held-out block
        in_excl = (r_target >= np.datetime64(ex_start)) & (r_target <= np.datetime64(ex_end))
        train = (~in_excl) & finite                                                # outside block and buffer
        tmo = pd.DatetimeIndex(r_target).month.values                              # target calendar month of each row
        X["base_rate"] = clim_cell[rw["r_i"], rw["r_j"]]                           # observed CDHW frequency 1951-1981, cell
        X["base_rate_month"] = clim_cm[tmo - 1, rw["r_i"], rw["r_j"]]              # ... cell x target calendar month
        Xtr, ytr, Xte, yte = X[train], y_all[train], X[test], y_all[test]
        print("  predictors: %d | train rows %d (event rate %.3f) | test rows %d (event rate %.3f)"
              % (X.shape[1], len(ytr), ytr.mean(), len(yte), yte.mean()))

        # 2) scaling and hyperparameter search on the training rows only
        t_step = time.time()
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        base = xgb.XGBClassifier(objective="binary:logistic", eval_metric="aucpr",
                                 scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(), n_jobs=N_JOBS, random_state=42)
        search = RandomizedSearchCV(base, PARAM_DIST, n_iter=40, scoring="neg_brier_score",
                                    cv=TimeSeriesSplit(n_splits=5), random_state=42, n_jobs=N_JOBS)
        search.fit(Xtr_s, ytr)                               # rows are in time order, as TimeSeriesSplit needs
        best = search.best_estimator_

        # 3) out-of-fold probabilities of the training rows: 5 contiguous time blocks, each predicted by a model trained on the
        #    others (same hyperparameters), the months within BUFFER_MONTHS of the predicted block are left out of its training
        tr_target = r_target[train]
        chunks = np.array_split(np.unique(tr_target), 5)
        oof = np.full(len(ytr), np.nan)
        for ch in chunks:
            in_ch = np.isin(tr_target, ch)
            lo = np.datetime64(pd.Timestamp(ch.min()) - pd.DateOffset(months=BUFFER_MONTHS))
            hi = np.datetime64(pd.Timestamp(ch.max()) + pd.DateOffset(months=BUFFER_MONTHS))
            inner_train = ~((tr_target >= lo) & (tr_target <= hi))
            m = clone(best).fit(Xtr_s[inner_train], ytr[inner_train])
            oof[in_ch] = m.predict_proba(Xtr_s[in_ch])[:, 1]

        # 4) global sigmoid calibration and threshold (fallback), final model of the fold
        platt = LogisticRegression().fit(logit(oof).reshape(-1, 1), ytr)
        p_oof = platt.predict_proba(logit(oof).reshape(-1, 1))[:, 1]
        thr = np.quantile(p_oof, 1 - ytr.mean())
        final = clone(best).fit(Xtr_s, ytr)

        # 5) class-wise calibration and threshold: classes of the 1951-1981 cell-month frequency (base_rate_month);
        #    the threshold of a class flags the 1951-1981 frequency of the class
        f_row = X["base_rate_month"].values
        cls_all = np.digitize(f_row, CLASS_EDGES, right=True)
        cls_tr, cls_te = cls_all[train], cls_all[test]
        platt_c, thr_c = {}, {}
        for c in range(len(CLASS_EDGES) + 1):
            idx = cls_tr == c
            if ytr[idx].sum() >= MIN_EVENTS_CLASS:
                platt_c[c] = LogisticRegression().fit(logit(oof[idx]).reshape(-1, 1), ytr[idx])
                p_c = platt_c[c].predict_proba(logit(oof[idx]).reshape(-1, 1))[:, 1]
                thr_c[c] = np.quantile(p_c, 1 - f_row[train][idx].mean())
            else:
                platt_c[c], thr_c[c] = platt, thr
        tick("  fold %d %s training, tuning, calibration" % (k, name), t_step)

        # 6) held-out block: final model -> raw probability -> class-wise calibration -> class-wise threshold
        p_raw = final.predict_proba(Xte_s)[:, 1]
        p_te, pred_te = np.zeros(len(p_raw)), np.zeros(len(p_raw), dtype=int)
        for c in range(len(CLASS_EDGES) + 1):
            idx = cls_te == c
            if idx.any():
                p_te[idx] = platt_c[c].predict_proba(logit(p_raw[idx]).reshape(-1, 1))[:, 1]
                pred_te[idx] = (p_te[idx] >= thr_c[c]).astype(int)
        preds[name].append(pd.DataFrame({
            "fold": k, "init": init_times[rw["r_init"][test]], "target": rw["target_times"][rw["r_init"][test]],
            "lat": lat[rw["r_i"][test]], "lon": lon[rw["r_j"][test]], "cdhw": yte,
            "proba": p_te, "proba_raw": p_raw, "pred": pred_te, "cell_class": cls_te,
            "clim_month": clim_cm[tmo[test] - 1, rw["r_i"][test], rw["r_j"][test]],
            "clim_season": clim_season[season_idx[tmo[test] - 1], rw["r_i"][test], rw["r_j"][test]]}))
        fold_info[name].append({"fold": k, "years": "%d-%d" % (y0, y1), "n_train": len(ytr), "n_test": len(yte),
                                "threshold": float(thr), "class_thresholds": json.dumps({int(c): float(v) for c, v in thr_c.items()}),
                                "best_params": json.dumps(search.best_params_)})
    tick("Fold %d total (all configurations)" % k, t_fold)

#%% Cell 5: save the predictions and the scores of every configuration (out of sample, pooled and per fold)
def scores(d):
    y, p, f = d["cdhw"].values, d["proba"].values, d["pred"].values
    bs = brier_score_loss(y, p)
    return {"N": len(y), "EventRate": y.mean(),
            "Precision": precision_score(y, f, zero_division=0), "Recall": recall_score(y, f, zero_division=0),
            "BS": bs, "BSS": 1 - bs / brier_score_loss(y, np.full(len(y), y.mean())),
            "BSS_vs_cellmonth_clim": 1 - bs / brier_score_loss(y, d["clim_month"].values),
            "BSS_vs_cellseason_clim": 1 - bs / brier_score_loss(y, d["clim_season"].values),
            "AUPRC": average_precision_score(y, p)}


pd.set_option("display.width", 200)
summary = []
for name, kind, K, _ in CONFIGS:
    pred_all = pd.concat(preds[name], ignore_index=True)
    pred_all.to_csv(os.path.join(OUT_DIRS[name], "rolling_origin_predictions.csv"), index=False)
    pd.DataFrame(fold_info[name]).to_csv(os.path.join(OUT_DIRS[name], "rolling_origin_fold_info.csv"), index=False)
    rows = [dict(Group="Pooled (all held-out years)", **scores(pred_all))]
    for kk in sorted(pred_all["fold"].unique()):
        rows.append(dict(Group="Fold %d (%s)" % (kk, fold_info[name][kk - 1]["years"]), **scores(pred_all[pred_all["fold"] == kk])))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIRS[name], "rolling_origin_scores.csv"), index=False)
    print("\n===== %s =====\n" % name, res.round(3).to_string(index=False))
    summary.append(dict(Config=name, **rows[0]))
summ = pd.DataFrame(summary).drop(columns="Group")
summ.to_csv(os.path.join(ROOT, "C_all_configs_summary.csv"), index=False)
print("\nPooled out-of-sample scores by configuration:\n", summ.round(3).to_string(index=False))

#%% Cell 6: runtime summary
rt = pd.DataFrame(runtime)
total = (time.time() - T0) / 60
rt.loc[len(rt)] = {"step": "TOTAL runtime", "minutes": round(total, 2)}
rt.to_csv(os.path.join(ROOT, "C_runtime.csv"), index=False)
print("\nTotal runtime: %.1f minutes (%.2f hours)" % (total, total / 60))
