from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_pca(df_columns, target_cols_to_drop, target_name, config):
    """
    Decide which columns go into PCA and which columns are kept as raw features.
    - Excludes target columns and cross-lags (already in target_cols_to_drop)
    - Optionally keeps AR term and seasonal sin/cos outside PCA.
    """
    pca_cfg = config.get("model", {}).get("pca", {})
    keep_ar = bool(pca_cfg.get("keep_ar", True))
    keep_seasonals = bool(pca_cfg.get("keep_seasonals", True))

    # seasonal columns (only if exist)
    seasonal_candidates = ["sin_cycle_1", "cos_cycle_1", "sin_cycle_2", "cos_cycle_2"]
    seasonal_cols = [c for c in seasonal_candidates if c in df_columns] if keep_seasonals else []

    # AR term (only if exist and requested)
    if keep_ar:
        ar_col = "headline_1m" if target_name == "Headline" else "core_1m"
        keep_cols = [ar_col] if ar_col in df_columns else []
    else:
        keep_cols = []

    keep_cols += seasonal_cols
    keep_cols = list(dict.fromkeys(keep_cols))  # unique preserve order

    # Candidate predictors: everything except dropped targets/cross-lags
    base_X_cols = [c for c in df_columns if c not in target_cols_to_drop]

    # PCA block: base_X_cols excluding keep_cols
    pca_cols = [c for c in base_X_cols if c not in keep_cols]

    return pca_cols, keep_cols


def choose_r_from_train_std(X_train_std, config):
    """
    Choose number of factors r based on config.
    Supported:
      - kaiser: eigenvalues > 1 (only valid if standardized)
      - fixed:  r_fixed
    """
    pca_cfg = config.get("model", {}).get("pca", {})
    max_factors = int(pca_cfg.get("max_factors", 10))
    r_method = str(pca_cfg.get("r_method", "kaiser")).lower()

    pca_all = PCA(n_components=min(max_factors, X_train_std.shape[1]))
    pca_all.fit(X_train_std)
    eigs = pca_all.explained_variance_

    if r_method == "fixed":
        r = int(pca_cfg.get("r_fixed", 3))
        r = max(1, min(r, X_train_std.shape[1], max_factors))
        return r

    # default: kaiser
    r = int(np.sum(eigs > 1.0))
    r = max(1, min(r, X_train_std.shape[1], max_factors))
    return r


def make_factor_features_time_safe(X_train, X_test, pca_cols, keep_cols, config):
    """
    Fit StandardScaler + PCA on TRAIN only, transform TRAIN+TEST.
    Returns X_train_final, X_test_final and fitted objects for debugging if needed.
    """
    # 1) Fit scaler on TRAIN PCA columns
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train[pca_cols])

    # 2) choose r and fit PCA on TRAIN only
    r = choose_r_from_train_std(X_train_std, config)
    pca = PCA(n_components=r)
    F_train = pca.fit_transform(X_train_std)

    # 3) transform TEST using train-fitted scaler+pca
    X_test_std = scaler.transform(X_test[pca_cols])
    F_test = pca.transform(X_test_std)

    factor_cols = [f"Factor_{i+1}" for i in range(r)]
    F_train_df = pd.DataFrame(F_train, index=X_train.index, columns=factor_cols)
    F_test_df = pd.DataFrame(F_test, index=X_test.index, columns=factor_cols)

    # 4) keep raw cols (AR + seasonals) outside PCA if requested
    kept_train = X_train[keep_cols].copy() if len(keep_cols) > 0 else pd.DataFrame(index=X_train.index)
    kept_test  = X_test[keep_cols].copy()  if len(keep_cols) > 0 else pd.DataFrame(index=X_test.index)

    # 5) final design
    X_train_final = pd.concat([kept_train, F_train_df], axis=1)
    X_test_final  = pd.concat([kept_test,  F_test_df], axis=1)

    return X_train_final, X_test_final, {"r": r, "pca_cols": pca_cols, "keep_cols": keep_cols, "scaler": scaler, "pca": pca}
