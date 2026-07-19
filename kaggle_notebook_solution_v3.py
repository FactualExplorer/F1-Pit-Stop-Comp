# %%
# =============================================================================
# KAGGLE NOTEBOOK V4 — FAST & DIVERSE
# =============================================================================
# PURPOSE: Generate predictions DIFFERENT from public notebooks so that
# blending this with your current 0.95440 pushes past 0.95465.
#
# RUNTIME: ~30-40 minutes on Kaggle GPU T4
#
# KEY DIFFERENCES FROM PUBLIC NOTEBOOKS:
#   - NormTyreLife proxy from original dataset median stint lengths
#   - Stint deviation features (overextension signals)
#   - Strategic pit window flags  
#   - External aggregation stats from original data
#   - Only LGB + XGB (no CatBoost → saves ~40% time)
#   - 2 seeds × 5 folds = 20 total trainings
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.linear_model import Ridge
from scipy.stats import rankdata
import xgboost as xgb
import lightgbm as lgb
import time, gc, os, glob

# %%
N_FOLDS = 5
SEEDS = [42, 2024]
EARLY_STOPPING = 200
np.random.seed(42)

if os.path.exists("/kaggle/input/playground-series-s6e5"):
    INPUT_DIR = "/kaggle/input/playground-series-s6e5"
    OUTPUT_DIR = "/kaggle/working"
    print("✅ Kaggle")
else:
    INPUT_DIR = "."
    OUTPUT_DIR = "."
    print("ℹ️ Local")

# Find original dataset
ORIG_DATA_PATH = None
for p in ["/kaggle/input/f1-strategy-dataset-pit-stop-prediction",
          "/kaggle/input/f1-strategy-dataset",
          "/kaggle/input/f1strategy-dataset-pit-stop-prediction"]:
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
print(f"  Orig data: {'✅' if ORIG_DATA_PATH else '⚠️ Not found'}")

USE_GPU = False
try:
    import torch
    USE_GPU = torch.cuda.is_available()
    if USE_GPU: print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    USE_GPU = os.system("nvidia-smi > /dev/null 2>&1") == 0
if not USE_GPU: print("⚠️ CPU mode")

# %%
# =============================================================================
# LOAD DATA
# =============================================================================
t_start = time.time()
print("\n" + "="*60 + "\nLOADING DATA\n" + "="*60)

train = pd.read_csv(f"{INPUT_DIR}/train.csv")
test = pd.read_csv(f"{INPUT_DIR}/test.csv")
print(f"Train: {train.shape}, Test: {test.shape}")

orig_full = None
if ORIG_DATA_PATH:
    for f in glob.glob(f"{ORIG_DATA_PATH}/*.csv"):
        try:
            tmp = pd.read_csv(f, nrows=2)
            if "Normalized_TyreLife" in tmp.columns and "PitNextLap" in tmp.columns:
                orig_full = pd.read_csv(f)
                print(f"Original: {orig_full.shape}")
                break
        except: pass

# %%
# Stint length reference
median_stint = {"HARD": 27, "MEDIUM": 19, "SOFT": 15, "INTERMEDIATE": 14, "WET": 7}
mean_stint = median_stint.copy()
median_stint_cr = {}

if orig_full is not None:
    print("  → Stint length stats...")
    si = orig_full.groupby(["Driver","Race","Year","Stint"]).agg(
        max_tl=("TyreLife","max"), compound=("Compound","first")).reset_index()
    median_stint = si.groupby("compound")["max_tl"].median().to_dict()
    mean_stint = si.groupby("compound")["max_tl"].mean().to_dict()
    median_stint_cr = si.groupby(["compound","Race"])["max_tl"].median().to_dict()
    print(f"    {median_stint}")

# Combine train + original
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

target = combined[target_col].values
test_ids = test["id"].values

combined["is_test"] = 0; test["is_test"] = 1
test[target_col] = -1; test["data_source"] = -1
if "NTL_orig" not in test.columns: test["NTL_orig"] = np.nan
if "NTL_orig" not in combined.columns: combined["NTL_orig"] = np.nan

df = pd.concat([combined, test], ignore_index=True)
print(f"Combined: {df.shape}, Target rate: {target.mean():.4f}")

# %%
# =============================================================================
# FEATURE ENGINEERING — ALL VECTORIZED, NO df.apply()
# =============================================================================
print("\n" + "="*60 + "\nFEATURE ENGINEERING\n" + "="*60)
t_fe = time.time()

cat_cols = ["Driver", "Compound", "Race"]
for c in cat_cols:
    df[c+"_le"] = LabelEncoder().fit_transform(df[c].astype(str))

cmp_order = {"WET":0, "INTERMEDIATE":1, "SOFT":2, "MEDIUM":3, "HARD":4}
df["Compound_ord"] = df["Compound"].map(cmp_order)

# --- NormTyreLife proxy (VECTORIZED — no df.apply!) ---
print("  → NormTL proxy (vectorized)...")
df["exp_stint"] = df["Compound"].map(median_stint).fillna(20)
df["NormTL_proxy"] = df["TyreLife"] / df["exp_stint"]

# Compound-Race expected stint (VECTORIZED via merge instead of apply)
if median_stint_cr:
    cr_df = pd.DataFrame([
        {"Compound": k[0], "Race": k[1], "exp_stint_cr": v}
        for k, v in median_stint_cr.items()
    ])
    df = df.merge(cr_df, on=["Compound", "Race"], how="left")
    df["exp_stint_cr"] = df["exp_stint_cr"].fillna(df["exp_stint"])
else:
    df["exp_stint_cr"] = df["exp_stint"]
df["NormTL_proxy_cr"] = df["TyreLife"] / df["exp_stint_cr"]

# Data-driven NormTyreLife
for gcols, nm in [
    (["Compound","Race"], "NTL_cr"), (["Compound"], "NTL_c"),
    (["Compound","Stint"], "NTL_cs"), (["Driver","Compound"], "NTL_dc")]:
    mx = df.groupby(gcols)["TyreLife"].transform("max")
    df[nm] = df["TyreLife"] / (mx + 1)

# Use original NTL where available
has_orig = df["NTL_orig"].notna()
df["NormTL_best"] = df["NormTL_proxy"]
if has_orig.sum() > 0:
    df.loc[has_orig, "NormTL_best"] = df.loc[has_orig, "NTL_orig"]

# --- Stint deviation ---
print("  → Stint deviation...")
df["StintDev"] = df["TyreLife"] - df["exp_stint"]
df["StintDev_cr"] = df["TyreLife"] - df["exp_stint_cr"]
df["IsOverext"] = (df["StintDev"] > 0).astype(np.int8)
df["StintDev_sq"] = df["StintDev"] ** 2
df["StintDev_abs"] = df["StintDev"].abs()
df["OverextRatio"] = df["TyreLife"] / (df["exp_stint"] + 1)
df["StintDev_mean"] = df["TyreLife"] - df["Compound"].map(mean_stint).fillna(20)

# --- Tyre features ---
print("  → Tyre features...")
df["TL_sq"] = df["TyreLife"] ** 2
df["TL_sqrt"] = np.sqrt(df["TyreLife"])
df["TL_log"] = np.log1p(df["TyreLife"])
df["Deg_rate"] = df["Cumulative_Degradation"] / (df["TyreLife"] + 1)
df["Deg_per_lap"] = df["Cumulative_Degradation"] / (df["LapNumber"] + 1)
df["Deg_sq"] = df["Cumulative_Degradation"] ** 2
df["Deg_abs"] = df["Cumulative_Degradation"].abs()
df["TL_x_Cmp"] = df["TyreLife"] * df["Compound_ord"]
df["Deg_x_Cmp"] = df["Cumulative_Degradation"] * df["Compound_ord"]
df["Stint_x_TL"] = df["Stint"] * df["TyreLife"]

# NormTL interactions
df["NTL_x_Deg"] = df["NormTL_best"] * df["Cumulative_Degradation"]
df["NTL_x_Cmp"] = df["NormTL_best"] * df["Compound_ord"]
df["NTL_x_RP"] = df["NormTL_best"] * df["RaceProgress"]
df["NTL_x_Pos"] = df["NormTL_best"] * df["Position"]
df["NTL_sq"] = df["NormTL_best"] ** 2
df["NTLp_x_Deg"] = df["NormTL_proxy"] * df["Cumulative_Degradation"]
df["NTLp_sq"] = df["NormTL_proxy"] ** 2

# StintDev interactions
df["SD_x_Deg"] = df["StintDev"] * df["Cumulative_Degradation"]
df["SD_x_Pos"] = df["StintDev"] * df["Position"]
df["SD_x_RP"] = df["StintDev"] * df["RaceProgress"]
df["DR_x_RP"] = df["Deg_rate"] * df["RaceProgress"]
df["DR_x_SD"] = df["Deg_rate"] * df["StintDev"]

# --- Race context ---
print("  → Race context...")
df["TotalLaps"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).clip(0, 200)
df["LapsRem"] = (df["TotalLaps"] - df["LapNumber"]).clip(0, 200)
df["TL_per_LR"] = df["TyreLife"] / (df["LapsRem"] + 1)
df["RP_x_TL"] = df["RaceProgress"] * df["TyreLife"]
df["RP_x_Stint"] = df["RaceProgress"] * df["Stint"]
df["RP_x_Deg"] = df["RaceProgress"] * df["Cumulative_Degradation"]
df["IsLate"] = (df["RaceProgress"] > 0.80).astype(np.int8)
df["IsEarly"] = (df["RaceProgress"] < 0.15).astype(np.int8)
df["RP_sq"] = df["RaceProgress"] ** 2

# Strategic flags
df["PitWindow"] = ((df["RaceProgress"]>0.20) & (df["RaceProgress"]<0.75) & (df["TyreLife"]>8)).astype(np.int8)
df["NoPitYet"] = ((df["Stint"]==1) & (df["RaceProgress"]>0.3)).astype(np.int8)
df["FinalLong"] = (df["IsLate"] * (df["NormTL_proxy"]>0.8)).astype(np.int8)
df["OverextInWin"] = (df["IsOverext"] * df["PitWindow"]).astype(np.int8)
df["JustPitted"] = (df["PitStop"]==1).astype(np.int8)

# --- Position ---
print("  → Position features...")
df["Pos_x_RP"] = df["Position"] * df["RaceProgress"]
df["Pos_x_TL"] = df["Position"] * df["TyreLife"]
df["PosChg_abs"] = df["Position_Change"].abs()
df["LTD_abs"] = df["LapTime_Delta"].abs()
df["LTD_sq"] = df["LapTime_Delta"] ** 2
df["Pos_inv"] = 1.0 / (df["Position"] + 1)
df["LT_x_TL"] = df["LapTime (s)"] * df["TyreLife"]
df["LTD_x_TL"] = df["LapTime_Delta"] * df["TyreLife"]
df["Pos_x_SD"] = df["Position"] * df["StintDev"]
df["LTD_x_NTL"] = df["LapTime_Delta"] * df["NormTL_proxy"]

# --- Categorical combos ---
print("  → Categoricals...")
df["Drv_Cmp"] = df["Driver"].astype(str) + "_" + df["Compound"].astype(str)
df["Drv_Race"] = df["Driver"].astype(str) + "_" + df["Race"].astype(str)
df["Race_Cmp"] = df["Race"].astype(str) + "_" + df["Compound"].astype(str)
df["Race_Yr"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
df["Drv_Stint"] = df["Driver"].astype(str) + "_" + df["Stint"].astype(str)
df["Cmp_Stint"] = df["Compound"].astype(str) + "_" + df["Stint"].astype(str)
df["Drv_Yr"] = df["Driver"].astype(str) + "_" + df["Year"].astype(str)
df["R_C_S"] = df["Race"].astype(str)+"_"+df["Compound"].astype(str)+"_"+df["Stint"].astype(str)
df["D_C_S"] = df["Driver"].astype(str)+"_"+df["Compound"].astype(str)+"_"+df["Stint"].astype(str)

combos = ["Drv_Cmp","Drv_Race","Race_Cmp","Race_Yr","Drv_Stint","Cmp_Stint","Drv_Yr","R_C_S","D_C_S"]
for c in combos:
    df[c+"_le"] = LabelEncoder().fit_transform(df[c].astype(str))
for c in cat_cols + combos:
    df[c+"_freq"] = df[c].map(df[c].value_counts(normalize=True)).astype(np.float32)

# --- KBins ---
print("  → KBins...")
for c in ["TyreLife","RaceProgress","LapTime (s)","Cumulative_Degradation",
           "NormTL_proxy","LapTime_Delta","Position","StintDev"]:
    v = df[c].values.reshape(-1,1)
    for nb in [10, 20]:
        df[f"{c}_qb{nb}"] = KBinsDiscretizer(
            n_bins=nb, encode="ordinal", strategy="quantile"
        ).fit_transform(v).astype(np.float32).ravel()

# --- Target encoding ---
print("  → Target encoding...")
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
        tr_df.iloc[vi, tr_df.columns.get_loc(cn)] = \
            tr_df.iloc[vi][c].map(mp["s"]).fillna(gmean).values
    mp_f = tr_df.groupby(c)[target_col].agg(["mean","count"])
    mp_f["s"] = (mp_f["mean"]*mp_f["count"] + gmean*20)/(mp_f["count"]+20)
    te_df[cn] = te_df[c].map(mp_f["s"]).fillna(gmean).values

df = pd.concat([tr_df, te_df], ignore_index=True)
del tr_df, te_df; gc.collect()

# --- Aggregation ---
print("  → Aggregation features...")
aggs = [
    ("Driver","TyreLife",["mean","std","max","min"]),
    ("Driver","Cumulative_Degradation",["mean","std"]),
    ("Driver","LapTime (s)",["mean","std"]),
    ("Driver","NormTL_proxy",["mean","std","max"]),
    ("Driver","StintDev",["mean","std"]),
    ("Compound","TyreLife",["mean","std","max"]),
    ("Compound","LapTime_Delta",["mean","std"]),
    ("Race","TyreLife",["mean","std"]),
    ("Race","LapTime (s)",["mean","std"]),
    ("Stint","TyreLife",["mean","std","max"]),
    ("Drv_Cmp","TyreLife",["mean","max"]),
    ("Race_Cmp","TyreLife",["mean","max"]),
    ("Cmp_Stint","TyreLife",["mean","max"]),
]
for g, v, fns in aggs:
    for fn in fns:
        df[f"{g}_{v}_{fn}"] = df[g].map(df.groupby(g)[v].agg(fn)).astype(np.float32)

# External aggs from original
if orig_full is not None:
    print("  → External original data stats...")
    for gc_, vc_, pf in [
        ("Driver","TyreLife","oD_TL"), ("Driver","LapTime (s)","oD_LT"),
        ("Driver","Cumulative_Degradation","oD_Deg"),
        ("Compound","TyreLife","oC_TL"), ("Race","TyreLife","oR_TL"),
        ("Race","LapTime (s)","oR_LT")]:
        ag = orig_full.groupby(gc_)[vc_].agg(["mean","std"]).add_prefix(f"{pf}_")
        df = df.merge(ag, left_on=gc_, right_index=True, how="left")
    df["TL_d_oDrv"] = df["TyreLife"] - df["oD_TL_mean"].fillna(df["TyreLife"].mean())
    df["TL_d_oRace"] = df["TyreLife"] - df["oR_TL_mean"].fillna(df["TyreLife"].mean())

# Diff from mean + ratios
print("  → Diff-from-mean + ratios...")
for g in ["Driver","Compound","Race"]:
    for v in ["TyreLife","LapTime (s)","NormTL_proxy"]:
        df[f"{v}_d_{g}"] = df[v] - df[g].map(df.groupby(g)[v].mean())

df["TL_r_maxD"] = df["TyreLife"] / (df["Driver_TyreLife_max"] + 1)
df["TL_r_maxC"] = df["TyreLife"] / (df["Compound_TyreLife_max"] + 1)
df["TL_r_maxS"] = df["TyreLife"] / (df["Stint_TyreLife_max"] + 1)

print(f"  ✅ Features done in {time.time()-t_fe:.0f}s")

# %%
# =============================================================================
# PREPARE
# =============================================================================
drop = (["id",target_col,"is_test","data_source","NTL_orig",
         "exp_stint","exp_stint_cr"] + cat_cols + combos)
feat = [c for c in df.columns if c not in drop]
print(f"\nFeatures: {len(feat)}")

X = np.nan_to_num(df[df["is_test"]==0][feat].values.astype(np.float32), nan=0, posinf=0, neginf=0)
Xt = np.nan_to_num(df[df["is_test"]==1][feat].values.astype(np.float32), nan=0, posinf=0, neginf=0)
fn = feat.copy()
print(f"X: {X.shape}, Xt: {Xt.shape}")
del df, train, test, combined; gc.collect()

# %%
# =============================================================================
# TRAINING — 2 seeds × 2 models × 5 folds = 20 trainings (~30 min)
# =============================================================================
all_oof, all_test = {}, {}
t0_train = time.time()

for si, SEED in enumerate(SEEDS):
    print(f"\n{'#'*50}")
    print(f"  SEED {SEED} ({si+1}/{len(SEEDS)}) — {(time.time()-t0_train)/60:.0f}min elapsed")
    print(f"{'#'*50}")
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

    # --- XGBoost ---
    print(f"\n  XGB s{SEED}")
    oof, tp, sc = np.zeros(len(target)), np.zeros(len(Xt)), []
    t0 = time.time()
    for f, (ti, vi) in enumerate(skf.split(X, target)):
        m = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="auc",
            max_depth=8, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.5, min_child_weight=10, gamma=0.1,
            reg_alpha=0.2, reg_lambda=1.5, n_estimators=3000,
            early_stopping_rounds=EARLY_STOPPING,
            tree_method="gpu_hist" if USE_GPU else "hist",
            device="cuda" if USE_GPU else "cpu",
            random_state=SEED, n_jobs=-1, verbosity=0)
        m.fit(X[ti], target[ti], eval_set=[(X[vi], target[vi])], verbose=False)
        vp = m.predict_proba(X[vi])[:,1]
        oof[vi] = vp; tp += m.predict_proba(Xt)[:,1] / N_FOLDS
        sc.append(roc_auc_score(target[vi], vp))
        del m; gc.collect()
    all_oof[f"xgb_s{SEED}"] = oof; all_test[f"xgb_s{SEED}"] = tp
    print(f"    AUC: {roc_auc_score(target,oof):.5f} ± {np.std(sc):.5f} [{time.time()-t0:.0f}s]")

    # --- LightGBM ---
    print(f"  LGB s{SEED}")
    oof, tp, sc = np.zeros(len(target)), np.zeros(len(Xt)), []
    t0 = time.time()
    for f, (ti, vi) in enumerate(skf.split(X, target)):
        dt = lgb.Dataset(X[ti], target[ti], feature_name=fn, free_raw_data=False)
        dv = lgb.Dataset(X[vi], target[vi], feature_name=fn, free_raw_data=False)
        m = lgb.train(dict(objective="binary", metric="auc", num_leaves=127,
            learning_rate=0.05, feature_fraction=0.5, bagging_fraction=0.8,
            bagging_freq=5, min_child_samples=20, min_gain_to_split=0.01,
            reg_alpha=0.2, reg_lambda=1.5,
            device="gpu" if USE_GPU else "cpu",
            random_state=SEED, n_jobs=-1, verbose=-1),
            dt, 3000, valid_sets=[dv],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)])
        vp = m.predict(X[vi])
        oof[vi] = vp; tp += m.predict(Xt) / N_FOLDS
        sc.append(roc_auc_score(target[vi], vp))
        del m, dt, dv; gc.collect()
    all_oof[f"lgb_s{SEED}"] = oof; all_test[f"lgb_s{SEED}"] = tp
    print(f"    AUC: {roc_auc_score(target,oof):.5f} ± {np.std(sc):.5f} [{time.time()-t0:.0f}s]")

print(f"\n⏱️ Training: {(time.time()-t0_train)/60:.0f} min total")

# %%
# =============================================================================
# ENSEMBLE
# =============================================================================
print("\n" + "="*60 + "\nENSEMBLE\n" + "="*60)

names = list(all_oof.keys())
for n in names: print(f"  {n}: {roc_auc_score(target, all_oof[n]):.5f}")

oof_s = np.column_stack([all_oof[n] for n in names])
test_s = np.column_stack([all_test[n] for n in names])

# Simple avg
savg_o = oof_s.mean(1); savg_t = test_s.mean(1)
print(f"\n  Simple avg: {roc_auc_score(target, savg_o):.5f}")

# Rank avg
rank_o = np.column_stack([rankdata(all_oof[n]) for n in names]).mean(1)
rank_t = np.column_stack([rankdata(all_test[n]) for n in names]).mean(1)
rank_t_norm = (rank_t-rank_t.min())/(rank_t.max()-rank_t.min()+1e-8)
print(f"  Rank avg:   {roc_auc_score(target, rank_o):.5f}")

# Type avg then grid search
types = {}
for n in names:
    t = n.split("_s")[0]
    types.setdefault(t, {"o":[],"t":[]})
    types[t]["o"].append(all_oof[n]); types[t]["t"].append(all_test[n])
to = {t: np.mean(d["o"],0) for t,d in types.items()}
tt = {t: np.mean(d["t"],0) for t,d in types.items()}
tnames = list(to.keys())
for t in tnames: print(f"  {t} avg: {roc_auc_score(target, to[t]):.5f}")

# Grid search 2-type weights
print("  Grid searching...")
best_a, best_w = -1, [0.5, 0.5]
ol = [to[t] for t in tnames]; tl = [tt[t] for t in tnames]
for w1 in np.arange(0, 1.01, 0.02):
    w2 = 1 - w1
    b = ol[0]*w1 + ol[1]*w2
    a = roc_auc_score(target, b)
    if a > best_a: best_a, best_w = a, [w1, w2]
print(f"  Grid: {best_a:.5f} — {dict(zip(tnames, best_w))}")
grid_t = tl[0]*best_w[0] + tl[1]*best_w[1]

# Ridge
ridge_o, ridge_t = np.zeros(len(target)), np.zeros(len(Xt))
for _,(ti,vi) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(oof_s,target)):
    r = Ridge(alpha=100, random_state=42)
    r.fit(oof_s[ti], target[ti])
    ridge_o[vi] = r.predict(oof_s[vi]); ridge_t += r.predict(test_s)/5
print(f"  Ridge: {roc_auc_score(target, ridge_o):.5f}")

# %%
# =============================================================================
# SAVE
# =============================================================================
print("\n" + "="*60 + "\nSAVING\n" + "="*60)

results = {
    "grid": (best_a, np.clip(grid_t, 1e-6, 1-1e-6)),
    "rank": (roc_auc_score(target,rank_o), np.clip(rank_t_norm, 1e-6, 1-1e-6)),
    "simple": (roc_auc_score(target,savg_o), np.clip(savg_t, 1e-6, 1-1e-6)),
    "ridge": (roc_auc_score(target,ridge_o), np.clip(ridge_t, 1e-6, 1-1e-6)),
}

best = max(results, key=lambda k: results[k][0])
for nm,(auc,p) in sorted(results.items(), key=lambda x:x[1][0], reverse=True):
    tag = " ★" if nm==best else ""
    fn_out = "submission.csv" if nm==best else f"submission_{nm}.csv"
    pd.DataFrame({"id": test_ids, "PitNextLap": p}).to_csv(f"{OUTPUT_DIR}/{fn_out}", index=False)
    print(f"  ✅ {fn_out:30s} AUC: {auc:.5f}{tag}")

print(f"\n  Total time: {(time.time()-t_start)/60:.0f} min")
print(f"  Best: {best} → {results[best][0]:.5f}")
print(f"\n  NOW BLEND THIS with your 0.95440 submission!")
print("="*60)
