# %%
# =============================================================================
# KAGGLE NOTEBOOK V6 — THE FINAL PUSH
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from scipy.stats import rankdata
from scipy.special import expit, logit as sp_logit
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import time, gc, os, glob

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================
N_FOLDS = 10
SEEDS = [42]
EARLY_STOPPING = 250
np.random.seed(42)

if os.path.exists("/kaggle/input/playground-series-s6e5"):
    INPUT_DIR = "/kaggle/input/playground-series-s6e5"
    OUTPUT_DIR = "/kaggle/working"
    print("✅ Running on Kaggle")
else:
    INPUT_DIR = "."
    OUTPUT_DIR = "."
    print("ℹ️ Running locally")

ORIG_DATA_PATH = None
for p in ["/kaggle/input/f1-strategy-dataset-pit-stop-prediction",
          "/kaggle/input/f1-strategy-dataset",
          "/kaggle/input/f1strategy-dataset-pit-stop-prediction",
          "/kaggle/input/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction"]:
    if os.path.exists(p) and glob.glob(f"{p}/*.csv"):
        ORIG_DATA_PATH = p; break
if ORIG_DATA_PATH is None:
    for d in glob.glob("/kaggle/input/*/"):
        for f in glob.glob(f"{d}*.csv"):
            try:
                if "Normalized_TyreLife" in pd.read_csv(f, nrows=2).columns:
                    ORIG_DATA_PATH = d.rstrip("/"); break
            except: pass
        if ORIG_DATA_PATH: break
if ORIG_DATA_PATH is None and os.path.exists("f1_strategy_dataset_v4.csv"):
    ORIG_DATA_PATH = "."
print(f"  Original Data: {'✅ Found' if ORIG_DATA_PATH else '⚠️ Not Found'}")

USE_GPU = False
try:
    import torch
    USE_GPU = torch.cuda.is_available()
    if USE_GPU: print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    USE_GPU = os.system("nvidia-smi > /dev/null 2>&1") == 0

# %%
# =============================================================================
# DATA LOADING & MERGING
# =============================================================================
t_start = time.time()
print("\n" + "="*60 + "\nLOADING DATA\n" + "="*60)

train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e5/train.csv")
test = pd.read_csv("/kaggle/input/competitions/playground-series-s6e5/test.csv")
print(f"Train: {train.shape}, Test: {test.shape}")

orig_full = None
if ORIG_DATA_PATH:
    for f in glob.glob(f"{ORIG_DATA_PATH}/*.csv"):
        try:
            tmp = pd.read_csv(f, nrows=2)
            if "Normalized_TyreLife" in tmp.columns and "PitNextLap" in tmp.columns:
                orig_full = pd.read_csv(f)
                break
        except: pass

median_stint = {"HARD": 27, "MEDIUM": 19, "SOFT": 15, "INTERMEDIATE": 14, "WET": 7}
mean_stint = median_stint.copy()
median_stint_cr = {}

if orig_full is not None:
    si = orig_full.groupby(["Driver","Race","Year","Stint"]).agg(
        max_tl=("TyreLife","max"), compound=("Compound","first")).reset_index()
    median_stint = si.groupby("compound")["max_tl"].median().to_dict()
    mean_stint = si.groupby("compound")["max_tl"].mean().to_dict()
    median_stint_cr = si.groupby(["compound","Race"])["max_tl"].median().to_dict()

target_col = "PitNextLap"
if orig_full is not None:
    orig_data = orig_full.copy()
    orig_ntl = orig_data["Normalized_TyreLife"].copy()
    common = [c for c in train.columns if c in orig_data.columns]
    orig_data = orig_data[common].copy()
    if "id" not in orig_data.columns:
        orig_data["id"] = range(len(train), len(train)+len(orig_data))
    orig_data["NTL_orig"] = orig_ntl.values
    train["data_source"] = 0; orig_data["data_source"] = 1
    combined = pd.concat([train, orig_data], ignore_index=True)
else:
    train["data_source"] = 0
    combined = train.copy()
    combined["NTL_orig"] = np.nan

test_ids = test["id"].values
combined["is_test"] = 0; test["is_test"] = 1
test[target_col] = -1; test["data_source"] = -1
if "NTL_orig" not in test.columns: test["NTL_orig"] = np.nan
if "NTL_orig" not in combined.columns: combined["NTL_orig"] = np.nan

df = pd.concat([combined, test], ignore_index=True)
df["orig_index"] = df.index

# %%
# =============================================================================
# V6 FEATURE ENGINEERING
# =============================================================================
print("\n" + "="*60 + "\nFEATURE ENGINEERING\n" + "="*60)
t_fe = time.time()

# --- 1. Base NTL Proxy ---
df["exp_stint"] = df["Compound"].map(median_stint).fillna(20)
df["NormTL_proxy"] = df["TyreLife"] / df["exp_stint"]
if median_stint_cr:
    cr_df = pd.DataFrame([{"Compound": k[0], "Race": k[1], "exp_stint_cr": v} for k, v in median_stint_cr.items()])
    df = df.merge(cr_df, on=["Compound", "Race"], how="left")
    df["exp_stint_cr"] = df["exp_stint_cr"].fillna(df["exp_stint"])
else:
    df["exp_stint_cr"] = df["exp_stint"]
df["NormTL_proxy_cr"] = df["TyreLife"] / df["exp_stint_cr"]

has_orig = df["NTL_orig"].notna()
df["NormTL_best"] = df["NormTL_proxy"]
if has_orig.sum() > 0:
    df.loc[has_orig, "NormTL_best"] = df.loc[has_orig, "NTL_orig"]

# --- 2. Temporal Lags ---
print("  → Temporal Lags")
df = df.sort_values(by=["Race", "Year", "Driver", "LapNumber"])
df["LapNumber_diff"] = df.groupby(["Race", "Year", "Driver"])["LapNumber"].diff()
valid_lag = df["LapNumber_diff"] == 1.0

df["TyreLife_diff"] = df.groupby(["Race", "Year", "Driver"])["TyreLife"].diff()
df.loc[~valid_lag, "TyreLife_diff"] = np.nan
df["CumDeg_diff"] = df.groupby(["Race", "Year", "Driver"])["Cumulative_Degradation"].diff()
df.loc[~valid_lag, "CumDeg_diff"] = np.nan
df["Pos_diff"] = df.groupby(["Race", "Year", "Driver"])["Position"].diff()
df.loc[~valid_lag, "Pos_diff"] = np.nan

# --- 3. Traffic Density ---
print("  → Traffic Density")
df = df.sort_values(by=["Race", "Year", "LapNumber"])
df["LT_bin"] = (df["LapTime (s)"] * 2).round() / 2
df["Traffic_0.5s"] = df.groupby(["Race", "Year", "LapNumber", "LT_bin"])["id"].transform("count") - 1

# --- 4. Synthetic Artifact Features ---
print("  → Artifact Digits")
df["LapTime_decimals"] = df["LapTime (s)"].astype(str).str.split('.').str[1].str.len().fillna(0).astype(np.float32)
df["CumDeg_decimals"] = df["Cumulative_Degradation"].astype(str).str.split('.').str[1].str.len().fillna(0).astype(np.float32)
df["LT_last_digit"] = df["LapTime (s)"].astype(str).str[-1].astype(np.float32).fillna(-1)

# Restore original order
df = df.sort_values(by="orig_index").reset_index(drop=True)
df.drop(columns=["orig_index"], inplace=True)

# Extract Target after restoring order
target = df[df["is_test"]==0][target_col].values

# --- 5. Interactions & Remaining Base Features ---
cat_cols = ["Driver", "Compound", "Race"]
for c in cat_cols:
    df[c+"_le"] = LabelEncoder().fit_transform(df[c].astype(str))

cmp_order = {"WET":0, "INTERMEDIATE":1, "SOFT":2, "MEDIUM":3, "HARD":4}
df["Compound_ord"] = df["Compound"].map(cmp_order)

df["StintDev"] = df["TyreLife"] - df["exp_stint"]
df["IsOverext"] = (df["StintDev"] > 0).astype(np.int8)
df["StintDev_sq"] = df["StintDev"] ** 2

df["TL_log"] = np.log1p(df["TyreLife"])
df["Deg_rate"] = df["Cumulative_Degradation"] / (df["TyreLife"] + 1)
df["Deg_per_lap"] = df["Cumulative_Degradation"] / (df["LapNumber"] + 1)
df["NTL_x_Deg"] = df["NormTL_best"] * df["Cumulative_Degradation"]
df["NTL_x_RP"] = df["NormTL_best"] * df["RaceProgress"]

df["TotalLaps"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).clip(0, 200)
df["LapsRem"] = (df["TotalLaps"] - df["LapNumber"]).clip(0, 200)
df["IsLate"] = (df["RaceProgress"] > 0.80).astype(np.int8)
df["PitWindow"] = ((df["RaceProgress"]>0.20) & (df["RaceProgress"]<0.75) & (df["TyreLife"]>8)).astype(np.int8)
df["NoPitYet"] = ((df["Stint"]==1) & (df["RaceProgress"]>0.3)).astype(np.int8)
df["JustPitted"] = (df["PitStop"]==1).astype(np.int8)

df["Pos_x_RP"] = df["Position"] * df["RaceProgress"]
df["PosChg_abs"] = df["Position_Change"].abs()
df["LTD_abs"] = df["LapTime_Delta"].abs()

combos = ["Drv_Cmp","Drv_Race","Race_Cmp","Race_Yr","Drv_Stint","Cmp_Stint"]
df["Drv_Cmp"] = df["Driver"].astype(str) + "_" + df["Compound"].astype(str)
df["Drv_Race"] = df["Driver"].astype(str) + "_" + df["Race"].astype(str)
df["Race_Cmp"] = df["Race"].astype(str) + "_" + df["Compound"].astype(str)
df["Race_Yr"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
df["Drv_Stint"] = df["Driver"].astype(str) + "_" + df["Stint"].astype(str)
df["Cmp_Stint"] = df["Compound"].astype(str) + "_" + df["Stint"].astype(str)

for c in combos:
    df[c+"_le"] = LabelEncoder().fit_transform(df[c].astype(str))
for c in cat_cols + combos:
    df[c+"_freq"] = df[c].map(df[c].value_counts(normalize=True)).astype(np.float32)

te_cols = cat_cols + combos + ["Year", "Stint"]
tr_df = df[df["is_test"]==0].copy()
te_df = df[df["is_test"]==1].copy()
skf_te = StratifiedKFold(5, shuffle=True, random_state=42)
gmean = target.mean()

for c in te_cols:
    cn = c+"_te"
    tr_df[cn] = 0.0; te_df[cn] = 0.0
    for _, (ti, vi) in enumerate(skf_te.split(tr_df, target)):
        tr = tr_df.iloc[ti]
        mp = tr.groupby(c)[target_col].agg(["mean","count"])
        mp["s"] = (mp["mean"]*mp["count"] + gmean*20)/(mp["count"]+20)
        tr_df.iloc[vi, tr_df.columns.get_loc(cn)] = tr_df.iloc[vi][c].map(mp["s"]).fillna(gmean).values
    mp_f = tr_df.groupby(c)[target_col].agg(["mean","count"])
    mp_f["s"] = (mp_f["mean"]*mp_f["count"] + gmean*20)/(mp_f["count"]+20)
    te_df[cn] = te_df[c].map(mp_f["s"]).fillna(gmean).values
df = pd.concat([tr_df, te_df], ignore_index=True)
del tr_df, te_df; gc.collect()

aggs = [
    ("Driver","TyreLife",["mean","std","max","min"]),
    ("Driver","Cumulative_Degradation",["mean","std"]),
    ("Compound","TyreLife",["mean","max"]),
    ("Compound","LapTime_Delta",["mean","std"]),
    ("Race","TyreLife",["mean","std"]),
    ("Stint","TyreLife",["mean","max"]),
]
for g, v, fns in aggs:
    for fn in fns:
        df[f"{g}_{v}_{fn}"] = df[g].map(df.groupby(g)[v].agg(fn)).astype(np.float32)

print(f"  ✅ Features engineered in {time.time()-t_fe:.0f}s")

# %%
# =============================================================================
# PREPARE MATRICES
# =============================================================================
drop = (["id",target_col,"is_test","data_source","NTL_orig","exp_stint","exp_stint_cr"] 
        + cat_cols + combos + ["LapNumber_diff", "LT_bin"])
feat = [c for c in df.columns if c not in drop]
print(f"\nFinal Features: {len(feat)}")

X = np.nan_to_num(df[df["is_test"]==0][feat].values.astype(np.float32), nan=0, posinf=0, neginf=0)
Xt = np.nan_to_num(df[df["is_test"]==1][feat].values.astype(np.float32), nan=0, posinf=0, neginf=0)
fn = feat.copy()
del df, train, test, combined; gc.collect()

# %%
# =============================================================================
# MODEL TRAINING (10 FOLDS, 1 SEED)
# =============================================================================
all_oof, all_test = {}, {}
t0_train = time.time()
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEEDS[0])

# --- 1. XGBoost ---
print(f"\n{'='*40}\n1. XGBoost\n{'='*40}")
oof, tp, sc = np.zeros(len(target)), np.zeros(len(Xt)), []
t0 = time.time()
for f, (ti, vi) in enumerate(skf.split(X, target)):
    m = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        max_depth=8, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.5, min_child_weight=10, gamma=0.1,
        reg_alpha=0.2, reg_lambda=1.5, n_estimators=8000,
        early_stopping_rounds=EARLY_STOPPING,
        tree_method="hist",
        device="cuda" if USE_GPU else "cpu",
        random_state=SEEDS[0], n_jobs=-1, verbosity=0)
    m.fit(X[ti], target[ti], eval_set=[(X[vi], target[vi])], verbose=False)
    vp = m.predict_proba(X[vi])[:,1]
    oof[vi] = vp; tp += m.predict_proba(Xt)[:,1] / N_FOLDS
    sc.append(roc_auc_score(target[vi], vp))
    del m; gc.collect()
all_oof["xgb"] = oof; all_test["xgb"] = tp
print(f"  → XGB AUC: {roc_auc_score(target,oof):.5f} ± {np.std(sc):.5f} [{time.time()-t0:.0f}s]")

# --- 2. LightGBM ---
print(f"\n{'='*40}\n2. LightGBM\n{'='*40}")
oof, tp, sc = np.zeros(len(target)), np.zeros(len(Xt)), []
t0 = time.time()
for f, (ti, vi) in enumerate(skf.split(X, target)):
    dt = lgb.Dataset(X[ti], target[ti], feature_name=fn, free_raw_data=False)
    dv = lgb.Dataset(X[vi], target[vi], feature_name=fn, free_raw_data=False)
    m = lgb.train(dict(objective="binary", metric="auc", num_leaves=127,
        learning_rate=0.02, feature_fraction=0.5, bagging_fraction=0.8,
        bagging_freq=5, min_child_samples=20, min_gain_to_split=0.01,
        reg_alpha=0.2, reg_lambda=1.5,
        device="gpu" if USE_GPU else "cpu",
        random_state=SEEDS[0], n_jobs=-1, verbose=-1),
        dt, 8000, valid_sets=[dv],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)])
    vp = m.predict(X[vi])
    oof[vi] = vp; tp += m.predict(Xt) / N_FOLDS
    sc.append(roc_auc_score(target[vi], vp))
    del m, dt, dv; gc.collect()
all_oof["lgb"] = oof; all_test["lgb"] = tp
print(f"  → LGB AUC: {roc_auc_score(target,oof):.5f} ± {np.std(sc):.5f} [{time.time()-t0:.0f}s]")

# --- 3. CatBoost ---
print(f"\n{'='*40}\n3. CatBoost\n{'='*40}")
oof, tp, sc = np.zeros(len(target)), np.zeros(len(Xt)), []
t0 = time.time()
for f, (ti, vi) in enumerate(skf.split(X, target)):
    cbp = dict(iterations=8000, depth=9, learning_rate=0.02,
        l2_leaf_reg=3.0, random_strength=1.0, eval_metric="AUC",
        random_seed=SEEDS[0], early_stopping_rounds=EARLY_STOPPING,
        verbose=0, task_type="GPU" if USE_GPU else "CPU")
    if USE_GPU:
        cbp["bootstrap_type"] = "Bernoulli"; cbp["subsample"] = 0.8
    else:
        cbp["subsample"] = 0.8; cbp["colsample_bylevel"] = 0.5
    m = cb.CatBoostClassifier(**cbp)
    m.fit(X[ti], target[ti], eval_set=(X[vi], target[vi]), verbose=0)
    vp = m.predict_proba(X[vi])[:,1]
    oof[vi] = vp; tp += m.predict_proba(Xt)[:,1] / N_FOLDS
    sc.append(roc_auc_score(target[vi], vp))
    del m; gc.collect()
all_oof["cb"] = oof; all_test["cb"] = tp
print(f"  → CB AUC: {roc_auc_score(target,oof):.5f} ± {np.std(sc):.5f} [{time.time()-t0:.0f}s]")

# %%
# =============================================================================
# ENSEMBLE & META-MODEL STACKING
# =============================================================================
names = list(all_oof.keys())
rank_o = np.column_stack([rankdata(all_oof[n]) for n in names]).mean(1)
rank_t = np.column_stack([rankdata(all_test[n]) for n in names]).mean(1)
rank_t_norm = (rank_t-rank_t.min())/(rank_t.max()-rank_t.min()+1e-8)

def apply_power(preds, power=1.1):
    p = np.clip(preds, 1e-8, 1-1e-8)
    return np.clip(expit(sp_logit(p) * power), 1e-6, 1-1e-6)

pp_rank_t = apply_power(rank_t_norm, power=1.1)
pd.DataFrame({"id": test_ids, "PitNextLap": pp_rank_t}).to_csv(f"{OUTPUT_DIR}/submission_v6.csv", index=False)
print(f"  ✅ Saved submission_v6.csv")
