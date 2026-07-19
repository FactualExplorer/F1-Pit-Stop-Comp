# %%
# =============================================================================
# KAGGLE NOTEBOOK V2 — Playground Series S6E5: F1 Pit Stop Prediction
# =============================================================================
# SETUP INSTRUCTIONS:
#   1. Go to kaggle.com/competitions/playground-series-s6e5 → "New Notebook"
#   2. Enable GPU: Settings → Accelerator → GPU T4 x2
#   3. Add the ORIGINAL dataset: "+ Add Input" → search "f1-strategy-dataset"
#      → Add "F1 Strategy Dataset | Pit Stop Prediction" by mexwell
#   4. Copy-paste this entire file and Run All
# =============================================================================
# WHAT'S NEW IN V2:
#   ✦ Original F1 dataset used for training (has Normalized_TyreLife!)
#   ✦ Reconstructed Normalized_TyreLife for synthetic data
#   ✦ Multi-seed ensembling (3 seeds × 3 models = 9 base models)
#   ✦ KBins discretization features
#   ✦ OOF Stacking with Ridge meta-model
#   ✦ Better hyperparameters tuned from V1 results
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
import catboost as cb
import time
import gc
import os
import glob

# %%
# =============================================================================
# CONFIG
# =============================================================================
N_FOLDS = 10
SEEDS = [42, 2024, 7777]  # Multi-seed for diversity
EARLY_STOPPING = 200

np.random.seed(42)

# Auto-detect environment
if os.path.exists("/kaggle/input/playground-series-s6e5"):
    INPUT_DIR = "/kaggle/input/playground-series-s6e5"
    OUTPUT_DIR = "/kaggle/working"
    print("✅ Running on Kaggle")
else:
    INPUT_DIR = "."
    OUTPUT_DIR = "."
    print("ℹ️  Running locally")

# Find original dataset
ORIG_DATA_PATH = None
possible_orig_paths = [
    "/kaggle/input/f1-strategy-dataset-pit-stop-prediction",
    "/kaggle/input/f1-strategy-dataset",
    "/kaggle/input/f1strategy-dataset-pit-stop-prediction",
    "/kaggle/input/datasets/factualexplorer/f1-strategy-dataset-v4/f1_strategy_dataset_v4.csv",
]
for p in possible_orig_paths:
    if os.path.exists(p):
        csv_files = glob.glob(f"{p}/*.csv")
        if csv_files:
            ORIG_DATA_PATH = p
            print(f"✅ Original F1 dataset found at: {p}")
            break

if ORIG_DATA_PATH is None:
    # Try to find it in any kaggle input directory
    for d in glob.glob("/kaggle/input/*/"):
        csv_files = glob.glob(f"{d}*.csv")
        for f in csv_files:
            try:
                tmp = pd.read_csv(f, nrows=2)
                if "Normalized_TyreLife" in tmp.columns:
                    ORIG_DATA_PATH = d.rstrip("/")
                    print(f"✅ Original F1 dataset found at: {ORIG_DATA_PATH}")
                    break
            except:
                pass
        if ORIG_DATA_PATH:
            break

if ORIG_DATA_PATH is None:
    print("⚠️  Original F1 dataset NOT found. Add it as input for best results!")
    print("   Go to: + Add Input → Search 'f1-strategy-dataset-pit-stop-prediction'")

# GPU detection
USE_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        USE_GPU = True
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    if os.system("nvidia-smi > /dev/null 2>&1") == 0:
        USE_GPU = True
        print("✅ GPU detected")
if not USE_GPU:
    print("⚠️  No GPU — using CPU")

# %%
# =============================================================================
# DATA LOADING
# =============================================================================
print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train = pd.read_csv(f"{INPUT_DIR}/train.csv")
test = pd.read_csv(f"{INPUT_DIR}/test.csv")

print(f"Synthetic train: {train.shape}")
print(f"Test: {test.shape}")

# Load original dataset if available
orig_data = None
if ORIG_DATA_PATH:
    # Find the CSV with Normalized_TyreLife
    for f in glob.glob(f"{ORIG_DATA_PATH}/*.csv"):
        try:
            tmp = pd.read_csv(f, nrows=2)
            if "Normalized_TyreLife" in tmp.columns and "PitNextLap" in tmp.columns:
                orig_data = pd.read_csv(f)
                print(f"Original dataset: {orig_data.shape} from {os.path.basename(f)}")
                print(f"  Has Normalized_TyreLife: ✅")
                break
        except:
            pass

# Align original data columns with synthetic
if orig_data is not None:
    # Keep Normalized_TyreLife for feature engineering before dropping
    orig_norm_tyre = orig_data["Normalized_TyreLife"].copy() if "Normalized_TyreLife" in orig_data.columns else None

    # Align columns — keep only columns that exist in train (plus PitNextLap)
    common_cols = [c for c in train.columns if c in orig_data.columns]
    missing_in_orig = [c for c in train.columns if c not in orig_data.columns and c != 'id']
    if missing_in_orig:
        print(f"  Columns in synthetic but not in original: {missing_in_orig}")

    orig_data = orig_data[common_cols].copy()

    # Add id column if missing
    if 'id' not in orig_data.columns:
        orig_data['id'] = range(len(train), len(train) + len(orig_data))

    # Re-add Normalized_TyreLife for feature engineering
    if orig_norm_tyre is not None:
        orig_data["Normalized_TyreLife_orig"] = orig_norm_tyre.values

    print(f"  Aligned original data: {orig_data.shape}")

# %%
# =============================================================================
# COMBINE DATA
# =============================================================================
target_col = "PitNextLap"

# Combine synthetic train + original data
if orig_data is not None and target_col in orig_data.columns:
    # Mark data source
    train["data_source"] = "synthetic"
    orig_data["data_source"] = "original"
    combined_train = pd.concat([train, orig_data], axis=0, ignore_index=True)
    print(f"Combined train (synthetic + original): {combined_train.shape}")
else:
    train["data_source"] = "synthetic"
    combined_train = train.copy()
    print(f"Train (synthetic only): {combined_train.shape}")

target = combined_train[target_col].values
train_ids = combined_train["id"].values
test_ids = test["id"].values

# Build full dataset for feature engineering
combined_train["is_test"] = 0
test["is_test"] = 1
test[target_col] = -1
test["data_source"] = "test"
if "Normalized_TyreLife_orig" not in test.columns:
    test["Normalized_TyreLife_orig"] = np.nan
if "Normalized_TyreLife_orig" not in combined_train.columns:
    combined_train["Normalized_TyreLife_orig"] = np.nan

df = pd.concat([combined_train, test], axis=0, ignore_index=True)
print(f"Full dataset: {df.shape}")
print(f"Target rate: {target.mean():.4f}")

# %%
# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

cat_cols = ["Driver", "Compound", "Race"]

# --- Label Encoding ---
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_le"] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# --- Compound ordinal ---
compound_order = {"WET": 0, "INTERMEDIATE": 1, "SOFT": 2, "MEDIUM": 3, "HARD": 4}
df["Compound_ordinal"] = df["Compound"].map(compound_order)

# --- CRITICAL: Reconstruct Normalized_TyreLife ---
print("  → Reconstructing Normalized_TyreLife...")
# Method 1: Per Compound × Race max TyreLife
max_tl_compound_race = df.groupby(["Compound", "Race"])["TyreLife"].transform("max")
df["NormTyreLife_compound_race"] = df["TyreLife"] / (max_tl_compound_race + 1)

# Method 2: Per Compound max TyreLife
max_tl_compound = df.groupby("Compound")["TyreLife"].transform("max")
df["NormTyreLife_compound"] = df["TyreLife"] / (max_tl_compound + 1)

# Method 3: Per Compound × Stint
max_tl_compound_stint = df.groupby(["Compound", "Stint"])["TyreLife"].transform("max")
df["NormTyreLife_compound_stint"] = df["TyreLife"] / (max_tl_compound_stint + 1)

# Method 4: Per Driver × Compound
max_tl_driver_compound = df.groupby(["Driver", "Compound"])["TyreLife"].transform("max")
df["NormTyreLife_driver_compound"] = df["TyreLife"] / (max_tl_driver_compound + 1)

# Method 5: Per Race × Year × Compound
max_tl_ryc = df.groupby(["Race", "Year", "Compound"])["TyreLife"].transform("max")
df["NormTyreLife_race_year_compound"] = df["TyreLife"] / (max_tl_ryc + 1)

# If we have the original Normalized_TyreLife, use it directly for training rows
if "Normalized_TyreLife_orig" in df.columns:
    has_orig = df["Normalized_TyreLife_orig"].notna()
    if has_orig.sum() > 0:
        print(f"    Using original Normalized_TyreLife for {has_orig.sum()} rows")
        # Use the original value where available, otherwise use best reconstruction
        df["NormTyreLife_best"] = df["NormTyreLife_compound_race"]
        df.loc[has_orig, "NormTyreLife_best"] = df.loc[has_orig, "Normalized_TyreLife_orig"]
    else:
        df["NormTyreLife_best"] = df["NormTyreLife_compound_race"]
else:
    df["NormTyreLife_best"] = df["NormTyreLife_compound_race"]

# --- Tyre Strategy Features ---
print("  → Tyre strategy features...")
df["TyreLife_sq"] = df["TyreLife"] ** 2
df["TyreLife_cb"] = df["TyreLife"] ** 3
df["TyreLife_sqrt"] = np.sqrt(df["TyreLife"])
df["TyreLife_log1p"] = np.log1p(df["TyreLife"])

df["Degradation_rate"] = df["Cumulative_Degradation"] / (df["TyreLife"] + 1)
df["Degradation_per_lap"] = df["Cumulative_Degradation"] / (df["LapNumber"] + 1)
df["Degradation_sq"] = df["Cumulative_Degradation"] ** 2
df["Degradation_abs"] = df["Cumulative_Degradation"].abs()

df["TyreLife_x_Compound"] = df["TyreLife"] * df["Compound_ordinal"]
df["Degradation_x_Compound"] = df["Cumulative_Degradation"] * df["Compound_ordinal"]
df["Stint_x_TyreLife"] = df["Stint"] * df["TyreLife"]
df["Stint_x_Compound"] = df["Stint"] * df["Compound_ordinal"]

# NormTyreLife interactions
df["NormTL_x_Degradation"] = df["NormTyreLife_best"] * df["Cumulative_Degradation"]
df["NormTL_x_Compound"] = df["NormTyreLife_best"] * df["Compound_ordinal"]
df["NormTL_x_RaceProgress"] = df["NormTyreLife_best"] * df["RaceProgress"]
df["NormTL_x_Position"] = df["NormTyreLife_best"] * df["Position"]
df["NormTL_sq"] = df["NormTyreLife_best"] ** 2

# --- Race Context Features ---
print("  → Race context features...")
df["LapsRemaining_approx"] = (1 - df["RaceProgress"]) * df["LapNumber"] / (df["RaceProgress"] + 1e-6)
df["LapsRemaining_approx"] = df["LapsRemaining_approx"].clip(0, 200)

df["RaceProgress_x_TyreLife"] = df["RaceProgress"] * df["TyreLife"]
df["RaceProgress_x_Stint"] = df["RaceProgress"] * df["Stint"]
df["RaceProgress_x_Degradation"] = df["RaceProgress"] * df["Cumulative_Degradation"]

df["IsLateRace"] = (df["RaceProgress"] > 0.80).astype(int)
df["IsEarlyRace"] = (df["RaceProgress"] < 0.15).astype(int)
df["IsMidRace"] = ((df["RaceProgress"] >= 0.3) & (df["RaceProgress"] <= 0.7)).astype(int)

df["RaceProgress_sq"] = df["RaceProgress"] ** 2
df["RaceProgress_cb"] = df["RaceProgress"] ** 3

# --- Position Features ---
print("  → Position features...")
df["Position_x_RaceProgress"] = df["Position"] * df["RaceProgress"]
df["Position_x_TyreLife"] = df["Position"] * df["TyreLife"]
df["Position_Change_abs"] = df["Position_Change"].abs()
df["LapTime_Delta_abs"] = df["LapTime_Delta"].abs()
df["LapTime_Delta_sq"] = df["LapTime_Delta"] ** 2

df["Position_sq"] = df["Position"] ** 2
df["Position_inv"] = 1.0 / (df["Position"] + 1)

df["LapTime_x_TyreLife"] = df["LapTime (s)"] * df["TyreLife"]
df["LapTime_x_RaceProgress"] = df["LapTime (s)"] * df["RaceProgress"]
df["LapTime_Delta_x_TyreLife"] = df["LapTime_Delta"] * df["TyreLife"]

# --- Combination Categoricals ---
print("  → Categorical combinations...")
df["Driver_Compound"] = df["Driver"].astype(str) + "_" + df["Compound"].astype(str)
df["Driver_Race"] = df["Driver"].astype(str) + "_" + df["Race"].astype(str)
df["Race_Compound"] = df["Race"].astype(str) + "_" + df["Compound"].astype(str)
df["Race_Year"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
df["Driver_Stint"] = df["Driver"].astype(str) + "_" + df["Stint"].astype(str)
df["Compound_Stint"] = df["Compound"].astype(str) + "_" + df["Stint"].astype(str)
df["Driver_Year"] = df["Driver"].astype(str) + "_" + df["Year"].astype(str)
df["Race_Compound_Stint"] = df["Race"].astype(str) + "_" + df["Compound"].astype(str) + "_" + df["Stint"].astype(str)

combo_cat_cols = ["Driver_Compound", "Driver_Race", "Race_Compound", "Race_Year",
                  "Driver_Stint", "Compound_Stint", "Driver_Year", "Race_Compound_Stint"]
for col in combo_cat_cols:
    le = LabelEncoder()
    df[col + "_le"] = le.fit_transform(df[col].astype(str))

# --- Frequency Encoding ---
print("  → Frequency encoding...")
for col in cat_cols + combo_cat_cols:
    freq = df[col].value_counts(normalize=True)
    df[col + "_freq"] = df[col].map(freq).astype(np.float32)

# %%
# --- KBins Discretization ---
print("  → KBins discretization...")
kbins_cols = ["TyreLife", "RaceProgress", "LapTime (s)", "Cumulative_Degradation",
              "NormTyreLife_best", "LapTime_Delta", "Position"]
for col in kbins_cols:
    vals = df[col].values.reshape(-1, 1)
    for n_bins in [10, 20]:
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        df[f"{col}_qbin{n_bins}"] = kbd.fit_transform(vals).astype(np.float32).ravel()

# %%
# --- Target Encoding (K-fold regularized) ---
print("  → Target encoding (K-fold regularized)...")
te_cols = cat_cols + combo_cat_cols + ["Year", "Stint"]

train_mask = df["is_test"] == 0
test_mask = df["is_test"] == 1
train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
global_mean = target.mean()
smoothing = 20

for col in te_cols:
    col_name = col + "_te"
    train_df[col_name] = 0.0
    test_df[col_name] = 0.0

    for fold_idx, (tr_idx, val_idx) in enumerate(skf_te.split(train_df, target)):
        tr_data = train_df.iloc[tr_idx]
        te_map = tr_data.groupby(col)[target_col].agg(["mean", "count"])
        te_map["smoothed"] = (te_map["mean"] * te_map["count"] + global_mean * smoothing) / (te_map["count"] + smoothing)
        train_df.iloc[val_idx, train_df.columns.get_loc(col_name)] = \
            train_df.iloc[val_idx][col].map(te_map["smoothed"]).fillna(global_mean).values

    te_map_full = train_df.groupby(col)[target_col].agg(["mean", "count"])
    te_map_full["smoothed"] = (te_map_full["mean"] * te_map_full["count"] + global_mean * smoothing) / (te_map_full["count"] + smoothing)
    test_df[col_name] = test_df[col].map(te_map_full["smoothed"]).fillna(global_mean).values

df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
del train_df, test_df
gc.collect()

# %%
# --- Aggregation Features ---
print("  → Aggregation features...")
agg_configs = [
    ("Driver", "TyreLife", ["mean", "std", "max", "min"]),
    ("Driver", "Cumulative_Degradation", ["mean", "std"]),
    ("Driver", "LapTime (s)", ["mean", "std"]),
    ("Driver", "NormTyreLife_best", ["mean", "std", "max"]),
    ("Compound", "TyreLife", ["mean", "std", "max"]),
    ("Compound", "LapTime_Delta", ["mean", "std"]),
    ("Compound", "NormTyreLife_best", ["mean", "std"]),
    ("Race", "TyreLife", ["mean", "std"]),
    ("Race", "LapTime (s)", ["mean", "std"]),
    ("Race", "NormTyreLife_best", ["mean", "std"]),
    ("Stint", "TyreLife", ["mean", "std", "max"]),
    ("Driver_Compound", "TyreLife", ["mean", "max"]),
    ("Race_Compound", "TyreLife", ["mean", "max"]),
    ("Driver_Compound", "NormTyreLife_best", ["mean", "max"]),
]

for grp_col, val_col, agg_funcs in agg_configs:
    for func in agg_funcs:
        new_col = f"{grp_col}_{val_col}_{func}"
        agg_map = df.groupby(grp_col)[val_col].agg(func)
        df[new_col] = df[grp_col].map(agg_map).astype(np.float32)

# Difference from group mean
print("  → Difference-from-mean features...")
for grp_col in ["Driver", "Compound", "Race"]:
    for val_col in ["TyreLife", "LapTime (s)", "NormTyreLife_best"]:
        mean_map = df.groupby(grp_col)[val_col].mean()
        df[f"{val_col}_diff_{grp_col}_mean"] = df[val_col] - df[grp_col].map(mean_map)

# Ratio features
print("  → Ratio features...")
df["TyreLife_ratio_max_driver"] = df["TyreLife"] / (df["Driver_TyreLife_max"] + 1)
df["TyreLife_ratio_max_compound"] = df["TyreLife"] / (df["Compound_TyreLife_max"] + 1)
df["TyreLife_ratio_max_stint"] = df["TyreLife"] / (df["Stint_TyreLife_max"] + 1)
df["NormTL_ratio_max_driver"] = df["NormTyreLife_best"] / (df["Driver_NormTyreLife_best_max"] + 0.01)

print("  ✅ Feature engineering complete!")

# %%
# =============================================================================
# PREPARE FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("PREPARING FEATURES")
print("=" * 70)

drop_cols = (
    ["id", target_col, "is_test", "data_source", "Normalized_TyreLife_orig"]
    + cat_cols + combo_cat_cols
)
feature_cols = [c for c in df.columns if c not in drop_cols]
print(f"Total features: {len(feature_cols)}")

train_feat = df[df["is_test"] == 0][feature_cols].values.astype(np.float32)
test_feat = df[df["is_test"] == 1][feature_cols].values.astype(np.float32)

print(f"Train: {train_feat.shape}")
print(f"Test:  {test_feat.shape}")

feature_names = feature_cols.copy()

del df, train, test, combined_train
gc.collect()

# %%
# =============================================================================
# MULTI-SEED MODEL TRAINING
# =============================================================================
# Train 3 models × 3 seeds = 9 base learners for maximum diversity

all_oof = {}  # name -> oof predictions
all_test = {}  # name -> test predictions

for seed_idx, SEED in enumerate(SEEDS):
    print(f"\n{'#' * 70}")
    print(f"# SEED {SEED} ({seed_idx+1}/{len(SEEDS)})")
    print(f"{'#' * 70}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # --- XGBoost ---
    print(f"\n  XGBOOST (seed={SEED})")
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 7,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "gpu_hist" if USE_GPU else "hist",
        "device": "cuda" if USE_GPU else "cpu",
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
    }

    oof = np.zeros(len(target))
    tpreds = np.zeros(len(test_feat))
    scores = []

    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_feat, target)):
        model = xgb.XGBClassifier(**xgb_params, n_estimators=5000, early_stopping_rounds=EARLY_STOPPING)
        model.fit(train_feat[tr_idx], target[tr_idx],
                  eval_set=[(train_feat[val_idx], target[val_idx])], verbose=False)
        val_pred = model.predict_proba(train_feat[val_idx])[:, 1]
        oof[val_idx] = val_pred
        tpreds += model.predict_proba(test_feat)[:, 1] / N_FOLDS
        scores.append(roc_auc_score(target[val_idx], val_pred))
        del model; gc.collect()

    name = f"xgb_s{SEED}"
    all_oof[name] = oof
    all_test[name] = tpreds
    print(f"    OOF AUC: {roc_auc_score(target, oof):.5f} (mean: {np.mean(scores):.5f} ± {np.std(scores):.5f}) [{time.time()-t0:.0f}s]")

    # --- LightGBM ---
    print(f"  LIGHTGBM (seed={SEED})")
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 63,
        "learning_rate": 0.03,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "device": "gpu" if USE_GPU else "cpu",
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    oof = np.zeros(len(target))
    tpreds = np.zeros(len(test_feat))
    scores = []

    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_feat, target)):
        dtrain = lgb.Dataset(train_feat[tr_idx], label=target[tr_idx], feature_name=feature_names, free_raw_data=False)
        dval = lgb.Dataset(train_feat[val_idx], label=target[val_idx], feature_name=feature_names, free_raw_data=False)
        model = lgb.train(lgb_params, dtrain, num_boost_round=5000, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)])
        val_pred = model.predict(train_feat[val_idx])
        oof[val_idx] = val_pred
        tpreds += model.predict(test_feat) / N_FOLDS
        scores.append(roc_auc_score(target[val_idx], val_pred))
        del model, dtrain, dval; gc.collect()

    name = f"lgb_s{SEED}"
    all_oof[name] = oof
    all_test[name] = tpreds
    print(f"    OOF AUC: {roc_auc_score(target, oof):.5f} (mean: {np.mean(scores):.5f} ± {np.std(scores):.5f}) [{time.time()-t0:.0f}s]")

    # --- CatBoost ---
    print(f"  CATBOOST (seed={SEED})")
    oof = np.zeros(len(target))
    tpreds = np.zeros(len(test_feat))
    scores = []

    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_feat, target)):
        cb_params = {
            "iterations": 5000, "depth": 8, "learning_rate": 0.03,
            "l2_leaf_reg": 3.0, "eval_metric": "AUC",
            "random_seed": SEED, "early_stopping_rounds": EARLY_STOPPING,
            "verbose": 0, "task_type": "GPU" if USE_GPU else "CPU",
        }
        if USE_GPU:
            cb_params["bootstrap_type"] = "Bernoulli"
            cb_params["subsample"] = 0.8
        else:
            cb_params["subsample"] = 0.8
            cb_params["colsample_bylevel"] = 0.6

        model = cb.CatBoostClassifier(**cb_params)
        model.fit(train_feat[tr_idx], target[tr_idx],
                  eval_set=(train_feat[val_idx], target[val_idx]), verbose=0)
        val_pred = model.predict_proba(train_feat[val_idx])[:, 1]
        oof[val_idx] = val_pred
        tpreds += model.predict_proba(test_feat)[:, 1] / N_FOLDS
        scores.append(roc_auc_score(target[val_idx], val_pred))
        del model; gc.collect()

    name = f"cb_s{SEED}"
    all_oof[name] = oof
    all_test[name] = tpreds
    print(f"    OOF AUC: {roc_auc_score(target, oof):.5f} (mean: {np.mean(scores):.5f} ± {np.std(scores):.5f}) [{time.time()-t0:.0f}s]")

# %%
# =============================================================================
# ENSEMBLE (Grid Search + Ridge Stacking)
# =============================================================================
print("\n" + "=" * 70)
print("ENSEMBLE BLENDING")
print("=" * 70)

model_names = list(all_oof.keys())
n_models = len(model_names)
print(f"  {n_models} base models: {model_names}")

# Print individual model scores
for name in model_names:
    print(f"    {name}: OOF AUC = {roc_auc_score(target, all_oof[name]):.5f}")

# Stack OOF and test predictions
oof_stack = np.column_stack([all_oof[n] for n in model_names])
test_stack = np.column_stack([all_test[n] for n in model_names])

# --- Method 1: Simple average of all models ---
simple_avg_oof = oof_stack.mean(axis=1)
simple_avg_auc = roc_auc_score(target, simple_avg_oof)
print(f"\n  Simple Average (all {n_models} models) OOF AUC: {simple_avg_auc:.5f}")

# --- Method 2: Average per model type, then blend ---
# Average across seeds for each model type
xgb_avg_oof = np.mean([all_oof[n] for n in model_names if "xgb" in n], axis=0)
lgb_avg_oof = np.mean([all_oof[n] for n in model_names if "lgb" in n], axis=0)
cb_avg_oof = np.mean([all_oof[n] for n in model_names if "cb" in n], axis=0)

xgb_avg_test = np.mean([all_test[n] for n in model_names if "xgb" in n], axis=0)
lgb_avg_test = np.mean([all_test[n] for n in model_names if "lgb" in n], axis=0)
cb_avg_test = np.mean([all_test[n] for n in model_names if "cb" in n], axis=0)

print(f"  XGB avg OOF AUC: {roc_auc_score(target, xgb_avg_oof):.5f}")
print(f"  LGB avg OOF AUC: {roc_auc_score(target, lgb_avg_oof):.5f}")
print(f"  CB  avg OOF AUC: {roc_auc_score(target, cb_avg_oof):.5f}")

# Grid search on seed-averaged models
print("  Grid searching optimal 3-model weights...")
t0 = time.time()
STEP = 0.02
best_grid_auc = -1
best_grid_w = np.array([1/3, 1/3, 1/3])

for w1 in np.arange(0.0, 1.0 + STEP, STEP):
    for w2 in np.arange(0.0, 1.0 - w1 + STEP, STEP):
        w3 = 1.0 - w1 - w2
        if w3 < -1e-9:
            continue
        w3 = max(w3, 0.0)
        blend = xgb_avg_oof * w1 + lgb_avg_oof * w2 + cb_avg_oof * w3
        auc = roc_auc_score(target, blend)
        if auc > best_grid_auc:
            best_grid_auc = auc
            best_grid_w = np.array([w1, w2, w3])

print(f"  Grid Search OOF AUC: {best_grid_auc:.5f}  ({time.time()-t0:.1f}s)")
print(f"    Weights → XGB: {best_grid_w[0]:.2f}, LGB: {best_grid_w[1]:.2f}, CB: {best_grid_w[2]:.2f}")

grid_test = xgb_avg_test * best_grid_w[0] + lgb_avg_test * best_grid_w[1] + cb_avg_test * best_grid_w[2]

# --- Method 3: Ridge Stacking ---
print("  Ridge stacking on OOF predictions...")
ridge_oof = np.zeros(len(target))
ridge_test = np.zeros(len(test_feat))
ridge_scores = []

skf_stack = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, val_idx) in enumerate(skf_stack.split(oof_stack, target)):
    ridge = Ridge(alpha=100, random_state=42)
    ridge.fit(oof_stack[tr_idx], target[tr_idx])
    ridge_oof[val_idx] = ridge.predict(oof_stack[val_idx])
    ridge_test += ridge.predict(test_stack) / 5
    ridge_scores.append(roc_auc_score(target[val_idx], ridge_oof[val_idx]))

ridge_auc = roc_auc_score(target, ridge_oof)
print(f"  Ridge Stacking OOF AUC: {ridge_auc:.5f}")

# --- Method 4: Rank Average ---
oof_ranks_all = np.column_stack([rankdata(all_oof[n]) for n in model_names])
rank_avg_oof = oof_ranks_all.mean(axis=1)
rank_avg_auc = roc_auc_score(target, rank_avg_oof)
print(f"  Rank Average OOF AUC: {rank_avg_auc:.5f}")

test_ranks_all = np.column_stack([rankdata(all_test[n]) for n in model_names])
rank_avg_test = test_ranks_all.mean(axis=1)

# %%
# =============================================================================
# SELECT BEST & GENERATE SUBMISSIONS
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSIONS")
print("=" * 70)

results = {
    "simple_avg": (simple_avg_auc, oof_stack.mean(axis=1), test_stack.mean(axis=1)),
    "grid_weighted": (best_grid_auc, xgb_avg_oof * best_grid_w[0] + lgb_avg_oof * best_grid_w[1] + cb_avg_oof * best_grid_w[2], grid_test),
    "ridge_stack": (ridge_auc, ridge_oof, ridge_test),
    "rank_avg": (rank_avg_auc, rank_avg_oof, rank_avg_test),
}

best_name = max(results, key=lambda k: results[k][0])
best_auc = results[best_name][0]
best_test = results[best_name][2]

# Normalize rank-based predictions
if "rank" in best_name:
    best_test = (best_test - best_test.min()) / (best_test.max() - best_test.min())

best_test = np.clip(best_test, 1e-6, 1 - 1e-6)

# Save primary submission
pd.DataFrame({"id": test_ids, "PitNextLap": best_test}).to_csv(f"{OUTPUT_DIR}/submission.csv", index=False)
print(f"  ✅ submission.csv ({best_name}, OOF AUC: {best_auc:.5f})")

# Save all alternative submissions
for name, (auc, _, tpreds) in results.items():
    if name == best_name:
        continue
    tp = tpreds.copy()
    if "rank" in name:
        tp = (tp - tp.min()) / (tp.max() - tp.min())
    tp = np.clip(tp, 1e-6, 1 - 1e-6)
    pd.DataFrame({"id": test_ids, "PitNextLap": tp}).to_csv(f"{OUTPUT_DIR}/submission_{name}.csv", index=False)
    print(f"  ✅ submission_{name}.csv (OOF AUC: {auc:.5f})")

# %%
# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("★ FINAL SUMMARY ★")
print("=" * 70)

top10_score = 0.95465

print(f"\n  Individual Models (OOF AUC):")
for name in model_names:
    print(f"    {name:15s}: {roc_auc_score(target, all_oof[name]):.5f}")

print(f"\n  Seed-Averaged Models:")
print(f"    XGBoost avg:   {roc_auc_score(target, xgb_avg_oof):.5f}")
print(f"    LightGBM avg:  {roc_auc_score(target, lgb_avg_oof):.5f}")
print(f"    CatBoost avg:  {roc_auc_score(target, cb_avg_oof):.5f}")

print(f"\n  Ensembles:")
for name, (auc, _, _) in results.items():
    marker = " ★" if name == best_name else ""
    print(f"    {name:18s}: {auc:.5f}{marker}")

print(f"\n  Best:        {best_name} → {best_auc:.5f}")
print(f"  Top-10:      {top10_score:.5f}")
print(f"  Gap:         {best_auc - top10_score:+.5f}")
print(f"  Features:    {len(feature_cols)}")

if best_auc >= top10_score:
    print("\n  🏆 TOP-10 LOOKS ACHIEVABLE!")
else:
    print(f"\n  📊 Submit all variants — LB often differs from OOF!")

print(f"\n  📁 {OUTPUT_DIR}/submission.csv (primary)")
print("=" * 70)
print("DONE!")
print("=" * 70)
