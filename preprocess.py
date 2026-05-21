"""
preprocess.py
=============
Template for the ML Assignment preprocessing pipeline.

Instructions
------------
1. Implement your preprocessing steps in the `preprocess()` function.
2. Do NOT change the input/output filenames or the overall structure.


       python preprocess.py

   It will read  X_test.csv  and write  X_test_preprocessed.csv
   in the same directory.

Notes
-----
- Any parameters you fitted on the training set (e.g. mean, std, min, max)
  must be hard-coded here — do not load external files.
- The output must have the same number of rows as the input.
"""

import pandas as pd
import numpy as np

# =============================================================================
# PREPROCESSING FUNCTION — implement your pipeline here
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to the raw feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw features loaded from X_test.csv.

    Returns
    -------
    pd.DataFrame
        Preprocessed features, ready for model inference.
    """

    df = df.copy()

    #   X_train = pd.read_csv('X_train.csv')
    #   x1_median = X_train['x1'].median()  # Sonuç: -0.006345
    #   x2_median = X_train['x2'].median()  # Sonuç: 29.723467
    #   x3_median = X_train['x3'].median()  # Sonuç: 2472.062064
    #   x4_median = X_train['x4'].median()  # Sonuç: 0.511433

    x1_median = -0.006345
    x2_median = 29.723467
    x3_median = 2472.062064
    x4_median = 0.511433

    medians = {
        'x1': x1_median,
        'x2': x2_median,
        'x3': x3_median,
        'x4': x4_median
    }
    
    # Test setindeki olası NaN değerleri, ilgili sütunun eğitim setindeki medyanı ile doldurulur.
    for col, val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
            
    # Not: XGBoost ağaç tabanlı bir algoritma olduğu için aykırı değerlere (outliers) karşı dirençlidir
    # ve veriyi ölçeklendirmeye (standardizasyon/normalizasyon) ihtiyaç duymaz.
    # Bu sebeple ekstra ölçeklendirme işlemi eklenmemiştir.

    return df


# =============================================================================
# ENTRY POINT — do not modify below this line
# =============================================================================

if __name__ == "__main__":
    INPUT_FILE  = "X_test.csv"
    OUTPUT_FILE = "X_test_preprocessed.csv"

    print(f"Reading {INPUT_FILE} ...")
    raw = pd.read_csv(INPUT_FILE)
    print(f"  Input shape : {raw.shape}")

    processed = preprocess(raw)
    print(f"  Output shape: {processed.shape}")

    processed.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}.")
