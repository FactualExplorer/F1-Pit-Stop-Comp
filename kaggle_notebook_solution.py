# %%
# =============================================================================
# KAGGLE NOTEBOOK — Playground Series S6E5: F1 Pit Stop Prediction
# =============================================================================
# How to use:
#   1. Go to kaggle.com/competitions/playground-series-s6e5
#   2. Click "New Notebook" (or open an existing one)
#   3. In notebook settings (right panel): Enable GPU (T4 x2 or P100)
#   4. Copy-paste this ENTIRE file into a single cell OR split at "# %%" markers
#   5. Run all cells — submission.csv will appear in the Output tab
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import rankdata
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import time
import gc
import os

# %%
# =============================================================================
# CONFIG
# =============================================================================
N_FOLDS = 10
SEED = 42
EARLY_STOPPING = 200

np.random.seed(SEED)

# Auto-detect environment: Kaggle vs local
if os.path.exists("/kaggle/input/playground-series-s6e5"):
    INPUT_DIR = "/kaggle/input/playground-series-s6e5"
    OUTPUT_DIR = "/kaggle/working"
    print("✅ Running on Kaggle — GPU should be enabled in notebook settings!")
else:
    INPUT_DIR = "."
    OUTPUT_DIR = "."
    print("ℹ️  Running locally")

# Detect GPU availability
USE_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        USE_GPU = True
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
except ImportError:
    # Try checking nvidia-smi
    if os.system("nvidia-smi > /dev/null 2>&1") == 0:
        USE_GPU = True
        print("✅ GPU detected via nvidia-smi")

if not USE_GPU:
    print("⚠️  No GPU detected — will use CPU (slower but works fine)")

# %%
# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train = pd.read_csv(f"{INPUT_DIR}/train.csv")
test = pd.read_csv(f"{INPUT_DIR}/test.csv")
sub = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")

target = train["PitNextLap"].values
train_ids = train["id"].values
test_ids = test["id"].values

# Combine for consistent feature engineering
train["is_test"] = 0
test["is_test"] = 1
test["PitNextLap"] = -1
df = pd.concat([train, test], axis=0, ignore_index=True)

print(f"Combined shape: {df.shape}")
print(f"Target distribution: {np.mean(target):.4f} positive rate")

# %%
# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

cat_cols = ["Driver", "Compound", "Race"]
num_cols = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change"
]

# --- Label Encoding ---
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_le"] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# --- Compound ordinal encoding ---
compound_order = {"WET": 0, "INTERMEDIATE": 1, "SOFT": 2, "MEDIUM": 3, "HARD": 4}
df["Compound_ordinal"] = df["Compound"].map(compound_order)

# --- 1A: Tyre Strategy Features ---
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

# --- 1B: Race Context Features ---
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

# --- 1C: Driver & Position Features ---
print("  → Driver & position features...")
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

# --- 1D: Combination Categoricals ---
print("  → Categorical combinations...")
df["Driver_Compound"] = df["Driver"].astype(str) + "_" + df["Compound"].astype(str)
df["Driver_Race"] = df["Driver"].astype(str) + "_" + df["Race"].astype(str)
df["Race_Compound"] = df["Race"].astype(str) + "_" + df["Compound"].astype(str)
df["Race_Year"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
df["Driver_Stint"] = df["Driver"].astype(str) + "_" + df["Stint"].astype(str)
df["Compound_Stint"] = df["Compound"].astype(str) + "_" + df["Stint"].astype(str)

combo_cat_cols = ["Driver_Compound", "Driver_Race", "Race_Compound", "Race_Year",
                  "Driver_Stint", "Compound_Stint"]
for col in combo_cat_cols:
    le = LabelEncoder()
    df[col + "_le"] = le.fit_transform(df[col].astype(str))

# --- 1E: Frequency Encoding ---
print("  → Frequency encoding...")
for col in cat_cols + combo_cat_cols:
    freq = df[col].value_counts(normalize=True)
    df[col + "_freq"] = df[col].map(freq).astype(np.float32)

# %%
# --- 1F: Target Encoding (K-fold regularized) ---
print("  → Target encoding (K-fold regularized)...")
te_cols = cat_cols + combo_cat_cols + ["Year", "Stint"]

train_mask = df["is_test"] == 0
test_mask = df["is_test"] == 1
train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
global_mean = target.mean()
smoothing = 20

for col in te_cols:
    col_name = col + "_te"
    train_df[col_name] = 0.0
    test_df[col_name] = 0.0

    # OOF target encoding for train
    for fold_idx, (tr_idx, val_idx) in enumerate(skf_te.split(train_df, target)):
        tr_data = train_df.iloc[tr_idx]
        te_map = tr_data.groupby(col)["PitNextLap"].agg(["mean", "count"])
        te_map["smoothed"] = (te_map["mean"] * te_map["count"] + global_mean * smoothing) / (te_map["count"] + smoothing)
        train_df.iloc[val_idx, train_df.columns.get_loc(col_name)] = \
            train_df.iloc[val_idx][col].map(te_map["smoothed"]).fillna(global_mean).values

    # Full target encoding for test
    te_map_full = train_df.groupby(col)["PitNextLap"].agg(["mean", "count"])
    te_map_full["smoothed"] = (te_map_full["mean"] * te_map_full["count"] + global_mean * smoothing) / (te_map_full["count"] + smoothing)
    test_df[col_name] = test_df[col].map(te_map_full["smoothed"]).fillna(global_mean).values

# Reassemble
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
del train_df, test_df
gc.collect()

# %%
# --- 1G: Aggregation Features ---
print("  → Aggregation features...")
agg_configs = [
    ("Driver", "TyreLife", ["mean", "std", "max"]),
    ("Driver", "Cumulative_Degradation", ["mean", "std"]),
    ("Driver", "LapTime (s)", ["mean", "std"]),
    ("Compound", "TyreLife", ["mean", "std", "max"]),
    ("Compound", "LapTime_Delta", ["mean", "std"]),
    ("Race", "TyreLife", ["mean", "std"]),
    ("Race", "LapTime (s)", ["mean", "std"]),
    ("Stint", "TyreLife", ["mean", "std", "max"]),
    ("Driver_Compound", "TyreLife", ["mean", "max"]),
    ("Race_Compound", "TyreLife", ["mean", "max"]),
]

for grp_col, val_col, agg_funcs in agg_configs:
    for func in agg_funcs:
        new_col = f"{grp_col}_{val_col}_{func}"
        agg_map = df.groupby(grp_col)[val_col].agg(func)
        df[new_col] = df[grp_col].map(agg_map).astype(np.float32)

# Difference from group mean
print("  → Difference-from-mean features...")
for grp_col in ["Driver", "Compound", "Race"]:
    mean_map = df.groupby(grp_col)["TyreLife"].mean()
    df[f"TyreLife_diff_{grp_col}_mean"] = df["TyreLife"] - df[grp_col].map(mean_map)

    mean_map2 = df.groupby(grp_col)["LapTime (s)"].mean()
    df[f"LapTime_diff_{grp_col}_mean"] = df["LapTime (s)"] - df[grp_col].map(mean_map2)

# --- 1H: Ratio Features ---
print("  → Ratio features...")
df["TyreLife_ratio_max_driver"] = df["TyreLife"] / (df["Driver_TyreLife_max"] + 1)
df["TyreLife_ratio_max_compound"] = df["TyreLife"] / (df["Compound_TyreLife_max"] + 1)
df["TyreLife_ratio_max_stint"] = df["TyreLife"] / (df["Stint_TyreLife_max"] + 1)

print("  ✅ Feature engineering complete!")

# %%
# =============================================================================
# PREPARE FINAL FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("PREPARING FEATURES")
print("=" * 70)

drop_cols = ["id", "PitNextLap", "is_test"] + cat_cols + combo_cat_cols
feature_cols = [c for c in df.columns if c not in drop_cols]
print(f"Total features: {len(feature_cols)}")

train_feat = df[df["is_test"] == 0][feature_cols].values.astype(np.float32)
test_feat = df[df["is_test"] == 1][feature_cols].values.astype(np.float32)

print(f"Train features shape: {train_feat.shape}")
print(f"Test features shape:  {test_feat.shape}")

feature_names = feature_cols.copy()

del df, train, test
gc.collect()

# %%
# =============================================================================
# MODEL 1: XGBOOST (10-fold CV)
# =============================================================================
print("\n" + "=" * 70)
print("TRAINING XGBOOST (10-fold CV)")
print("=" * 70)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

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

oof_xgb = np.zeros(len(target))
test_preds_xgb = np.zeros(len(test_feat))
xgb_scores = []

t0 = time.time()
for fold, (tr_idx, val_idx) in enumerate(skf.split(train_feat, target)):
    X_tr, X_val = train_feat[tr_idx], train_feat[val_idx]
    y_tr, y_val = target[tr_idx], target[val_idx]

    model = xgb.XGBClassifier(
        **xgb_params,
        n_estimators=5000,
        early_stopping_rounds=EARLY_STOPPING,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_xgb[val_idx] = val_pred
    test_preds_xgb += model.predict_proba(test_feat)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_val, val_pred)
    xgb_scores.append(fold_auc)
    print(f"  Fold {fold+1:2d} | AUC: {fold_auc:.5f} | Best iter: {model.best_iteration}")

    del model, X_tr, X_val
    gc.collect()

xgb_oof_auc = roc_auc_score(target, oof_xgb)
print(f"\n  ★ XGBoost OOF AUC: {xgb_oof_auc:.5f} (mean: {np.mean(xgb_scores):.5f} ± {np.std(xgb_scores):.5f})")
print(f"  Time: {time.time() - t0:.1f}s")

# %%
# =============================================================================
# MODEL 2: LIGHTGBM (10-fold CV)
# =============================================================================
print("\n" + "=" * 70)
print("TRAINING LIGHTGBM (10-fold CV)")
print("=" * 70)

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

oof_lgb = np.zeros(len(target))
test_preds_lgb = np.zeros(len(test_feat))
lgb_scores = []

t0 = time.time()
for fold, (tr_idx, val_idx) in enumerate(skf.split(train_feat, target)):
    X_tr, X_val = train_feat[tr_idx], train_feat[val_idx]
    y_tr, y_val = target[tr_idx], target[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, free_raw_data=False)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_val)
    oof_lgb[val_idx] = val_pred
    test_preds_lgb += model.predict(test_feat) / N_FOLDS

    fold_auc = roc_auc_score(y_val, val_pred)
    lgb_scores.append(fold_auc)
    print(f"  Fold {fold+1:2d} | AUC: {fold_auc:.5f} | Best iter: {model.best_iteration}")

    del model, dtrain, dval, X_tr, X_val
    gc.collect()

lgb_oof_auc = roc_auc_score(target, oof_lgb)
print(f"\n  ★ LightGBM OOF AUC: {lgb_oof_auc:.5f} (mean: {np.mean(lgb_scores):.5f} ± {np.std(lgb_scores):.5f})")
print(f"  Time: {time.time() - t0:.1f}s")

# %%
# =============================================================================
# MODEL 3: CATBOOST (10-fold CV)
# =============================================================================
print("\n" + "=" * 70)
print("TRAINING CATBOOST (10-fold CV)")
print("=" * 70)

oof_cb = np.zeros(len(target))
test_preds_cb = np.zeros(len(test_feat))
cb_scores = []

t0 = time.time()
for fold, (tr_idx, val_idx) in enumerate(skf.split(train_feat, target)):
    X_tr, X_val = train_feat[tr_idx], train_feat[val_idx]
    y_tr, y_val = target[tr_idx], target[val_idx]

    cb_params = {
        "iterations": 5000,
        "depth": 8,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        "eval_metric": "AUC",
        "random_seed": SEED,
        "early_stopping_rounds": EARLY_STOPPING,
        "verbose": 0,
        "task_type": "GPU" if USE_GPU else "CPU",
    }
    if USE_GPU:
        # GPU mode: use Bernoulli bootstrap (Bayesian not supported on GPU)
        # colsample_bylevel is also not supported on GPU
        cb_params["bootstrap_type"] = "Bernoulli"
        cb_params["subsample"] = 0.8
    else:
        cb_params["subsample"] = 0.8
        cb_params["colsample_bylevel"] = 0.6

    model = cb.CatBoostClassifier(**cb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        verbose=0,
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_cb[val_idx] = val_pred
    test_preds_cb += model.predict_proba(test_feat)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_val, val_pred)
    cb_scores.append(fold_auc)
    print(f"  Fold {fold+1:2d} | AUC: {fold_auc:.5f} | Best iter: {model.best_iteration_}")

    del model, X_tr, X_val
    gc.collect()

cb_oof_auc = roc_auc_score(target, oof_cb)
print(f"\n  ★ CatBoost OOF AUC: {cb_oof_auc:.5f} (mean: {np.mean(cb_scores):.5f} ± {np.std(cb_scores):.5f})")
print(f"  Time: {time.time() - t0:.1f}s")

# %%
# =============================================================================
# ENSEMBLE BLENDING (fast grid search — finishes in ~30s)
# =============================================================================
print("\n" + "=" * 70)
print("ENSEMBLE BLENDING")
print("=" * 70)

oof_preds = np.column_stack([oof_xgb, oof_lgb, oof_cb])
test_preds = np.column_stack([test_preds_xgb, test_preds_lgb, test_preds_cb])

# --- Simple Average ---
simple_avg = oof_preds.mean(axis=1)
simple_avg_auc = roc_auc_score(target, simple_avg)
print(f"  Simple Average OOF AUC:      {simple_avg_auc:.5f}")

# --- Grid Search for Optimal Weights ---
# With 3 models, grid search at step=0.02 gives ~1,326 combos — fast & deterministic
print("  Searching optimal weights (grid search)...")
t0 = time.time()
STEP = 0.02
best_weighted_auc = -1
opt_weights = np.array([1/3, 1/3, 1/3])

for w1 in np.arange(0.0, 1.0 + STEP, STEP):
    for w2 in np.arange(0.0, 1.0 - w1 + STEP, STEP):
        w3 = 1.0 - w1 - w2
        if w3 < -1e-9:
            continue
        w3 = max(w3, 0.0)
        blend = oof_xgb * w1 + oof_lgb * w2 + oof_cb * w3
        auc = roc_auc_score(target, blend)
        if auc > best_weighted_auc:
            best_weighted_auc = auc
            opt_weights = np.array([w1, w2, w3])

opt_blend = (oof_preds * opt_weights).sum(axis=1)
opt_blend_auc = best_weighted_auc
print(f"  Optimized Weighted OOF AUC:  {opt_blend_auc:.5f}  ({time.time()-t0:.1f}s)")
print(f"    Weights → XGB: {opt_weights[0]:.2f}, LGB: {opt_weights[1]:.2f}, CB: {opt_weights[2]:.2f}")

# --- Rank Average ---
oof_ranks = np.column_stack([rankdata(oof_xgb), rankdata(oof_lgb), rankdata(oof_cb)])
rank_avg = oof_ranks.mean(axis=1)
rank_avg_auc = roc_auc_score(target, rank_avg)
print(f"  Rank Average OOF AUC:        {rank_avg_auc:.5f}")

# --- Grid Search for Optimal Rank Weights ---
print("  Searching optimal rank weights (grid search)...")
t0 = time.time()
best_rank_auc = -1
rank_opt_weights = np.array([1/3, 1/3, 1/3])

oof_rank_xgb = rankdata(oof_xgb)
oof_rank_lgb = rankdata(oof_lgb)
oof_rank_cb = rankdata(oof_cb)

for w1 in np.arange(0.0, 1.0 + STEP, STEP):
    for w2 in np.arange(0.0, 1.0 - w1 + STEP, STEP):
        w3 = 1.0 - w1 - w2
        if w3 < -1e-9:
            continue
        w3 = max(w3, 0.0)
        blend = oof_rank_xgb * w1 + oof_rank_lgb * w2 + oof_rank_cb * w3
        auc = roc_auc_score(target, blend)
        if auc > best_rank_auc:
            best_rank_auc = auc
            rank_opt_weights = np.array([w1, w2, w3])

rank_opt_blend = (oof_ranks * rank_opt_weights).sum(axis=1)
rank_opt_blend_auc = best_rank_auc
print(f"  Optimized Rank Blend OOF AUC:{rank_opt_blend_auc:.5f}  ({time.time()-t0:.1f}s)")
print(f"    Weights → XGB: {rank_opt_weights[0]:.2f}, LGB: {rank_opt_weights[1]:.2f}, CB: {rank_opt_weights[2]:.2f}")

# %%
# =============================================================================
# SELECT BEST ENSEMBLE & GENERATE SUBMISSIONS
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSIONS")
print("=" * 70)

test_ranks = np.column_stack([rankdata(test_preds_xgb), rankdata(test_preds_lgb), rankdata(test_preds_cb)])

ensemble_results = {
    "simple_avg": (simple_avg_auc, test_preds.mean(axis=1)),
    "opt_weighted": (opt_blend_auc, (test_preds * opt_weights).sum(axis=1)),
    "rank_avg": (rank_avg_auc, test_ranks.mean(axis=1)),
    "rank_opt": (rank_opt_blend_auc, (test_ranks * rank_opt_weights).sum(axis=1)),
}

best_name = max(ensemble_results, key=lambda k: ensemble_results[k][0])
best_auc = ensemble_results[best_name][0]
final_test_preds = ensemble_results[best_name][1]

# Normalize rank-based predictions to [0, 1]
if "rank" in best_name:
    final_test_preds = (final_test_preds - final_test_preds.min()) / (final_test_preds.max() - final_test_preds.min())

final_test_preds = np.clip(final_test_preds, 1e-6, 1 - 1e-6)

# --- Primary submission (best ensemble) ---
submission = pd.DataFrame({"id": test_ids, "PitNextLap": final_test_preds})
submission.to_csv(f"{OUTPUT_DIR}/submission.csv", index=False)
print(f"  ✅ submission.csv ({best_name}, OOF AUC: {best_auc:.5f})")

# --- Also save optimized weighted blend ---
opt_test = np.clip((test_preds * opt_weights).sum(axis=1), 1e-6, 1 - 1e-6)
pd.DataFrame({"id": test_ids, "PitNextLap": opt_test}).to_csv(
    f"{OUTPUT_DIR}/submission_opt_weighted.csv", index=False)
print(f"  ✅ submission_opt_weighted.csv (OOF AUC: {opt_blend_auc:.5f})")

# --- Also save rank optimized blend ---
rank_opt_test = (test_ranks * rank_opt_weights).sum(axis=1)
rank_opt_test = (rank_opt_test - rank_opt_test.min()) / (rank_opt_test.max() - rank_opt_test.min())
rank_opt_test = np.clip(rank_opt_test, 1e-6, 1 - 1e-6)
pd.DataFrame({"id": test_ids, "PitNextLap": rank_opt_test}).to_csv(
    f"{OUTPUT_DIR}/submission_rank_opt.csv", index=False)
print(f"  ✅ submission_rank_opt.csv (OOF AUC: {rank_opt_blend_auc:.5f})")

# --- Individual model submissions ---
for name, preds in [("xgb", test_preds_xgb), ("lgb", test_preds_lgb), ("cb", test_preds_cb)]:
    pd.DataFrame({"id": test_ids, "PitNextLap": np.clip(preds, 1e-6, 1-1e-6)}).to_csv(
        f"{OUTPUT_DIR}/submission_{name}.csv", index=False)
print(f"  ✅ submission_xgb.csv, submission_lgb.csv, submission_cb.csv")

# %%
# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("★ FINAL SUMMARY ★")
print("=" * 70)

top10_score = 0.95465
top1_score = 0.95493

print(f"""
  Individual Models (OOF AUC):
    XGBoost:   {xgb_oof_auc:.5f}
    LightGBM:  {lgb_oof_auc:.5f}
    CatBoost:  {cb_oof_auc:.5f}

  Ensembles (OOF AUC):
    Simple Average:      {simple_avg_auc:.5f}
    Optimized Weighted:  {opt_blend_auc:.5f}
    Rank Average:        {rank_avg_auc:.5f}
    Optimized Rank:      {rank_opt_blend_auc:.5f}

  Best Ensemble:  {best_name} → {best_auc:.5f}
  Top-10 target:  {top10_score:.5f}
  Gap to top-10:  {best_auc - top10_score:+.5f}

  Total features: {len(feature_cols)}
""")

if best_auc >= top10_score:
    print("  🏆 OOF score suggests TOP-10 is achievable!")
else:
    print("  📊 Submit anyway — public LB may differ from OOF!")

print(f"""
  📁 Files generated in {OUTPUT_DIR}/:
     → submission.csv              (primary — submit this first)
     → submission_opt_weighted.csv (alternative blend)
     → submission_rank_opt.csv     (rank-based blend)
     → submission_xgb/lgb/cb.csv   (individual models)

  💡 Tip: You get 10 submissions/day on Kaggle.
     Submit the primary first, then try alternatives.
""")
print("=" * 70)
print("DONE! Click 'Submit to Competition' on the Output tab →")
print("=" * 70)
