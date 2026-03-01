from pyparsing import alphas
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict

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


def make_factor_features_time_safe(X_train, X_test, pca_cols, keep_cols, config, forecast_date=None, target_name=None, h=None, top_k=5, pca_bundle=None):
    """
    fit StandardScaler+ PCA on traindata, transform train+test"""
    if pca_bundle is None:
        #initialize scaler for feature normalization
        scaler= StandardScaler()
        #fit and transform pca columns using train data
        X_train_std=scaler.fit_transform(X_train[pca_cols])
        #determine number of factors r using config logic
        r=choose_r_from_train_std(X_train_std, config)
        #initialize pca with selected components
        pca= PCA(n_components=r)
        #get factors for train set
        F_train= pca.fit_transform(X_train_std)
        #sign stabilization for interpretability
        for i in range(pca.components_.shape[0]):
            #check if first loading is negative
            if pca.components_[i, 0] < 0:
                #flip component signs
                pca.components_[i, :]*= -1
                #flip factor signs to match
                F_train[:, i]*= -1
    else:
        #extract scaler from bundle
        scaler= pca_bundle["scaler"]
        #extract fitted pca object
        pca=pca_bundle["pca"]
        #get number of components
        r=pca_bundle["r"]

        #transform train data using stored scaler
        X_train_std= scaler.transform(X_train[pca_cols])
        #project standardized data onto existing factors
        F_train= pca.transform(X_train_std)
    #apply the same scaling and PCA transformation to the test set
    X_test_std= scaler.transform(X_test[pca_cols])
    F_test= pca.transform(X_test_std)
    #capture loading matrix (variables x factors)
    loadings=pd.DataFrame(pca.components_.T, index=pca_cols, columns=[f"Factor_{i+1}" for i in range(r)])
    #def output path for components of the factors
    out_path = Path("Results/Factor_Summaries/Factor_Summary_bvar_independent_niw_PCA.csv")
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
    kept_test  = X_test[keep_cols].copy() if len(keep_cols)> 0 else pd.DataFrame(index=X_test.index)
    #raw features and the new PCA factors together for the final model input
    X_train_final= pd.concat([kept_train, F_train_df], axis=1)
    X_test_final =pd.concat([kept_test,  F_test_df], axis=1)
    #bundle to return:
    bundle = {"r": r, "pca_cols": pca_cols, "keep_cols": keep_cols, "scaler": scaler, "pca": pca,}
    return X_train_final, X_test_final, bundle


#-----------------------
#Robust Linear Model Fitting (In-Sample)
#-----------------------
def fit_enet_mean_and_residuals(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, h: int, pub_lag: int= 2):
    #convert train features to numeric
    X_train= X_train.apply(pd.to_numeric, errors="coerce")
    #convert test features to numeric
    X_test= X_test.apply(pd.to_numeric, errors="coerce")
    #drop missing target values
    y_train= y_train.dropna()
    #align feature matrix with cleaned targets
    X_train= X_train.loc[y_train.index]
    #get sample size for cv split logic
    n_samples= len(y_train)
    #check for sufficient samples or zero variance
    if n_samples < 30 or float(y_train.std())== 0.0:
        #return original target and zero mean if insufficient
        return y_train.copy(), 0.0
    #determine cv splits based on sample count
    if n_samples < 50:
        n_splits= 2
    elif n_samples < 100:
        n_splits= 3
    else:
        n_splits= 5
    #calculate gap for time series cross validation
    cv_gap= h + pub_lag
    #initialize tscv object
    tscv= TimeSeriesSplit(n_splits=n_splits, gap=cv_gap)
    #setup alpha grid for elastic net
    alphas= np.logspace(-3, 2, 50)
    #setup l1 ratio grid
    l1_ratio= [0.1, 0.5, 0.7, 0.9, 0.95]
    #define preprocessing and model pipeline
    pipeline= Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("enet", ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas, cv=tscv, n_jobs=-1, max_iter=100_000, selection='random', tol=1e-3))])
    #fit linear pipeline
    try:
        pipeline.fit(X_train, y_train)
    except Exception:
        #return fallback if fit fails
        return y_train.copy(), 0.0
    #get in-sample predictions
    mean_train= pipeline.predict(X_train)
    #get out-of-sample prediction point
    mean_test= float(pipeline.predict(X_test)[0])
    #calculate residuals for training
    y_resid_train= y_train - mean_train
    #check if residuals reduced variance
    if y_resid_train.std() > y_train.std():
        #revert to original if linear part adds noise
        return y_train.copy(), 0.0
    #return cleaned series of residuals
    return pd.Series(y_resid_train, index=y_train.index), mean_test