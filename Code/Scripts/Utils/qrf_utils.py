from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.base import clone

#PCA
#-----------------------
def get_pca(df_columns, target_cols_to_drop, target_name, config):
    """decide which columns go into PCA and which columns are kept as raw features"""
    #get info from features
    pca_cfg= config.get("model", {}).get("pca", {})
    #check if we should keep the autoregressive (AR) term 
    keep_ar= bool(pca_cfg.get("keep_ar", True))
    #check if we should keep seasonal features 
    keep_seasonals=bool(pca_cfg.get("keep_seasonals", True))
    #seasonal columns
    seasonal_candidates= ["sin_cycle_1", "cos_cycle_1", "sin_cycle_2", "cos_cycle_2"]
    seasonal_cols= [c for c in seasonal_candidates if c in df_columns] if keep_seasonals else []
    #AR term: pick headline or core depending on what we are forecasting
    if keep_ar:
        keep_cols= ["headline_1m", "core_1m"]
    else:
        keep_cols= []
    #combine AR and seasonal columns into one list of features to protect from PCA
    keep_cols+= seasonal_cols
    #make sure we don't have duplicates and keep the order the same
    keep_cols=list(dict.fromkeys(keep_cols)) 
    #get all potential predictors by removing targets and cross-lags
    base_X_cols= [c for c in df_columns if c not in target_cols_to_drop]
    #columns to be compressed: everything left over that isn't in the keep list
    pca_cols = [c for c in base_X_cols if c not in keep_cols]
    return pca_cols, keep_cols


def choose_r_from_train_std(X_train_std, config):
    """look up how many factors we want from the config"""
    #get config data
    pca_cfg= config.get("model", {}).get("pca", {})
    #set a cap on factors so the model doesn't overfit
    max_factors= int(pca_cfg.get("max_factors", 10))
    #method: either 'kaiser' (auto) or 'fixed' -> can compare
    r_method= str(pca_cfg.get("r_method", "kaiser")).lower()
    #run a preliminary PCA to see the eigenvalues/variance
    pca_all= PCA(n_components=min(max_factors, X_train_std.shape[1]))
    pca_all.fit(X_train_std)
    eigs= pca_all.explained_variance_
    #if fixed: just use the number from the config but stay within data limits
    if r_method== "fixed":
        r= int(pca_cfg.get("r_fixed", 3))
        r= max(1, min(r, X_train_std.shape[1], max_factors))
        return r
    #default (kaiser): only keep factors where the eigenvalue is greater than 1
    r= int(np.sum(eigs > 1.0))
    # again, make sure r is at least 1 and doesn't exceed our max settings
    r= max(1, min(r, X_train_std.shape[1], max_factors))
    return r


def make_factor_features_time_safe(X_train, X_test, pca_cols, keep_cols, config, forecast_date=None, target_name=None, h=None, top_k=5):
    """
    fit StandardScaler + PCA on traindata, transform train+test"""
    #setup and fit the scaler on the training data only (avoiding data leakage)
    scaler= StandardScaler()
    X_train_std= scaler.fit_transform(X_train[pca_cols])
    #determine the number of factors and fit the PCA on training data
    r= choose_r_from_train_std(X_train_std, config)  #call fct to get r
    pca = PCA(n_components=r) #get pca
    F_train=pca.fit_transform(X_train_std)  #fit train data w pca
    #do sign check for linear part: if factor sign flips between recursive windows, linear coeffs jump wildly
    for i in range(pca.components_.shape[0]):
        #if first element vec is negative, flip the sign
        if pca.components_[i, 0]< 0:
            pca.components_[i, :]*= -1
            F_train[:, i] *=-1   #also sign of the scores flip
    #apply the same scaling and PCA transformation to the test set
    X_test_std= scaler.transform(X_test[pca_cols])
    F_test= pca.transform(X_test_std)
    #capture loading matrix (variables x factors)
    loadings=pd.DataFrame(pca.components_.T, index=pca_cols, columns=[f"Factor_{i+1}" for i in range(r)])
    #def output path for components of the factors
    out_path = Path("Results/Factor_Summaries/Factor_Summary_BVAR_indep_niw.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    #initialize dict to get summary what factor contains what
    factor_details = []
    #loop through cols and get the top k features in the factor
    for factor in loadings.columns:
        #get top K by absolute weight
        top_vars=loadings[factor].abs().sort_values(ascending=False).head(top_k)
        for var_name, abs_val in top_vars.items():
            # Get the real weight (signed) to know if it's positive or negative correlation
            actual_weight = loadings.loc[var_name, factor]
            factor_details.append({"Date": forecast_date, "Target": target_name, "Horizon": h, "Factor": factor, "Variable": var_name, "Weight": round(actual_weight, 4)})
    #define df for summary of the features in the factor
    pd.DataFrame(factor_details).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    # reate names for the new factors (Factor_1, Factor_2, etc.)
    factor_cols = [f"Factor_{i+1}" for i in range(r)]
    #convert the numpy arrays back to dataframes with the original dates
    F_train_df= pd.DataFrame(F_train, index=X_train.index, columns=factor_cols)
    F_test_df= pd.DataFrame(F_test, index=X_test.index, columns=factor_cols)
    #grab the raw columns (AR + seasonals) that didn't want to compress
    kept_train= X_train[keep_cols].copy() if len(keep_cols)> 0 else pd.DataFrame(index=X_train.index)
    kept_test  = X_test[keep_cols].copy()  if len(keep_cols)> 0 else pd.DataFrame(index=X_test.index)
    #raw features and the new PCA factors together for the final model input
    X_train_final= pd.concat([kept_train, F_train_df], axis=1)
    X_test_final =pd.concat([kept_test,  F_test_df], axis=1)
    return X_train_final, X_test_final, {"r": r, "pca_cols": pca_cols, "keep_cols": keep_cols, "scaler": scaler, "pca": pca}

#linear features
#----------------------------------

def generate_linear_feature_oof(df, target_col, target_cols_to_drop, h, config, 
                                use_pca=False, window_size=120, min_train=40):
    """
    Generates Linear_Pred using strictly past data (Walk-Forward).
    Supports PCA feature reduction to match the QRF input.
    """
    # 1. Setup Data
    X_full = df.drop(columns=target_cols_to_drop)
    y_full = df[target_col]
    preds = pd.Series(index=df.index, dtype=float)

    # 2. Setup PCA Config (if used)
    if use_pca:
        # Determine which columns are PCA vs Keep once
        pca_cols, keep_cols = get_pca(df_columns=X_full.columns, 
                                      target_cols_to_drop=[], # Already dropped
                                      target_name=None, 
                                      config=config)
    else:
        # If not using PCA, we treat all columns as "keep" (features)
        pca_cols = []
        keep_cols = list(X_full.columns)

    # 3. Setup Model
    # ElasticNetCV automatically handles scaling internally better if we don't scale Y, 
    # but for X we must scale.
    n_splits = 5
    if h >= 12:
        model = ElasticNetCV(l1_ratio=[0.5, 0.7, 0.9], alphas=[50, 100, 500], cv=n_splits, n_jobs=-1)
    else:
        model = ElasticNetCV(l1_ratio=[0.1, 0.5], alphas=[0.1, 1.0, 10.0, 50.0], cv=n_splits, n_jobs=-1)
    
    scaler = StandardScaler()

    print(f"Generating Linear Features for {target_col} (PCA={use_pca})...")

    # 4. Iterate through time
    for i in range(len(df)):
        
        # Define indices
        train_end_idx = i - h
        
        # Check history availability
        if train_end_idx < min_train:
            preds.iloc[i] = 0.0
            continue

        # Define Rolling Window
        train_start_idx = max(0, train_end_idx - window_size)
        
        # Slice Data
        X_train_raw = X_full.iloc[train_start_idx : train_end_idx + 1]
        y_train = y_full.iloc[train_start_idx : train_end_idx + 1]
        X_test_raw = X_full.iloc[[i]]
        
        # Handle NaNs in Target
        valid_mask = ~y_train.isna()
        y_train_clean = y_train[valid_mask]
        X_train_clean = X_train_raw[valid_mask]
        
        # Safety Check for Cross-Validation
        if len(y_train_clean) < (n_splits + 2) or y_train_clean.std() == 0:
            preds.iloc[i] = 0.0
            continue

        try:
            # --- FEATURE ENGINEERING BLOCK ---
            if use_pca:
                # A. Handle "Keep" Columns (AR, Seasonals)
                if len(keep_cols) > 0:
                    X_train_keep = X_train_clean[keep_cols].values
                    X_test_keep = X_test_raw[keep_cols].values
                else:
                    X_train_keep = np.empty((len(X_train_clean), 0))
                    X_test_keep = np.empty((len(X_test_raw), 0))

                # B. Handle PCA Columns
                if len(pca_cols) > 0:
                    # Scale PCA columns (Fit on Train, Transform Test)
                    X_train_pca = scaler.fit_transform(X_train_clean[pca_cols])
                    X_test_pca = scaler.transform(X_test_raw[pca_cols])
                    
                    # Determine 'r' (dynamic number of factors)
                    # We reuse your existing function logic
                    r = choose_r_from_train_std(X_train_pca, config)
                    
                    # Fit PCA
                    pca = PCA(n_components=r)
                    F_train = pca.fit_transform(X_train_pca)
                    
                    # Sign Flip Logic (Crucial for stability!)
                    # We manually replicate your logic here to avoid the CSV writing side-effect
                    for k in range(pca.components_.shape[0]):
                        if pca.components_[k, 0] < 0:
                            pca.components_[k, :] *= -1
                            F_train[:, k] *= -1
                    
                    # Transform Test
                    F_test = pca.transform(X_test_pca)
                    
                    # Concatenate Keep + Factors
                    X_train_final = np.hstack([X_train_keep, F_train])
                    X_test_final = np.hstack([X_test_keep, F_test])
                else:
                    X_train_final = X_train_keep
                    X_test_final = X_test_keep

            else:
                # NO PCA: Just scale everything
                X_train_final = scaler.fit_transform(X_train_clean)
                X_test_final = scaler.transform(X_test_raw)

            # --- MODEL FITTING BLOCK ---
            m = clone(model)
            m.fit(X_train_final, y_train_clean)
            preds.iloc[i] = m.predict(X_test_final)[0]

        except Exception as e:
            # Catch singular matrices or other numerical issues
            preds.iloc[i] = 0.0

    preds = preds.fillna(0.0)
    return preds