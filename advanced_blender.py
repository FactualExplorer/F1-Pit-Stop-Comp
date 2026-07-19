import pandas as pd
import numpy as np
from scipy.stats import rankdata
from scipy.special import logit, expit
import glob
import sys
import os

def logit_rank_blend(csv_paths, weights=None, power=1.1, asym_gating=True):
    """
    Blends submission CSVs using Logit-Rank technique and asymmetric gating.
    """
    print(f"Loading {len(csv_paths)} files...")
    dfs = [pd.read_csv(p) for p in csv_paths]
    
    if weights is None:
        # Give slightly more weight to higher scoring files if we can guess from names
        weights = []
        for p in csv_paths:
            if "0.95454" in p or "top" in p.lower():
                weights.append(1.2)
            else:
                weights.append(1.0)
                
    weights = np.array(weights) / np.sum(weights)
    print(f"Weights used: {weights}")
    
    # Extract probabilities
    preds = [df['PitNextLap'].values for df in dfs]
    
    # 1. Rank normalize each prediction (scale 0 to 1)
    ranked_preds = [rankdata(p) / len(p) for p in preds]
    
    # 2. Convert to logit space
    clip_eps = 1e-6
    logit_preds = [logit(np.clip(p, clip_eps, 1-clip_eps)) for p in ranked_preds]
    
    # 3. Weighted average in logit space
    blended_logit = np.zeros_like(logit_preds[0])
    for w, lp in zip(weights, logit_preds):
        blended_logit += w * lp
        
    # 4. Asymmetric Min-Max Gating 
    # Pit stops are the minority class. 
    # If a model is highly confident (e.g., >0.9), we shouldn't mute it completely by averaging.
    if asym_gating:
        raw_max = np.max(preds, axis=0)
        raw_min = np.min(preds, axis=0)
        
        # We find where consensus is weak but one model is very confident
        high_conf_mask = raw_max > 0.95
        low_conf_mask = raw_min < 0.05
        
        # Convert raw max/min to logit space
        logit_max = logit(np.clip(raw_max, clip_eps, 1-clip_eps))
        logit_min = logit(np.clip(raw_min, clip_eps, 1-clip_eps))
        
        # Apply gating: pull blended logit towards the extreme if confidence is high
        blended_logit = np.where(high_conf_mask, blended_logit * 0.7 + logit_max * 0.3, blended_logit)
        blended_logit = np.where(low_conf_mask, blended_logit * 0.7 + logit_min * 0.3, blended_logit)
        
    # 5. Convert back to probability space using sigmoid (expit)
    # Apply a power law stretch (power > 1) to push predictions towards 0 and 1
    blended_prob = expit(blended_logit * power)
    
    # Create final submission
    sub = dfs[0].copy()
    sub['PitNextLap'] = blended_prob
    return sub

if __name__ == "__main__":
    files = sys.argv[1:]
    
    if not files:
        print("Please provide the CSV files as arguments.")
        print("Example: python advanced_blender.py top_1.csv top_2.csv top_3.csv my_best.csv")
        sys.exit(1)
        
    print(f"Blending {len(files)} files: {files}")
    
    # Run blender
    sub = logit_rank_blend(files, power=1.15, asym_gating=True)
    
    out_name = "advanced_blend_submission.csv"
    sub.to_csv(out_name, index=False)
    print(f"✅ Saved blended submission to {out_name}")
