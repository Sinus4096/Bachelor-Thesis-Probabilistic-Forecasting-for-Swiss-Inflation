from pyparsing import alphas
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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

#linear features
#----------------------------------
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict

def _feasible_tscv(n: int, gap: int, max_splits: int = 5):
    # conservative test_size
    test_size = max(6, min(12, n // 6))
    # feasibility: n >= (n_splits+1)*test_size + n_splits*gap
    max_possible = int((n - test_size) // (test_size + gap))
    n_splits = max(2, min(max_splits, max_possible))
    if n_splits < 2:
        return None
    return TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)

def fit_enet_mean_and_residuals(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    h: int,
    pub_lag: int = 2,
    max_splits: int = 5,
):
    """
    Returns:
      y_resid_train: cross-fitted residuals on training window (Series)
      mean_test: scalar mean prediction at X_test (float)
      mean_train_cf: cross-fitted mean predictions on training window (Series)
    """
    # force numeric and align
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    # drop NaNs in y; align X
    y_train = y_train.dropna()
    X_train = X_train.loc[y_train.index]

    if len(y_train) < 40 or y_train.std() == 0:
        # not enough signal -> fallback mean=0, residual=y
        mean_test = 0.0
        mean_train_cf = pd.Series(0.0, index=y_train.index)
        return (y_train - mean_train_cf), mean_test, mean_train_cf

    gap = h + pub_lag
    cv = _feasible_tscv(len(y_train), gap=gap, max_splits=max_splits)

    # If CV is not feasible, fallback to 2 splits with tiny test_size and gap,
    # otherwise just do simple fit and accept in-sample preds
    alphas = np.logspace(-4, 1.5, 60)
    l1_ratio = [0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 0.95]

    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("enet", ElasticNetCV(
            l1_ratio=l1_ratio,
            alphas=alphas,
            cv=cv if cv is not None else 3,
            n_jobs=-1,
            max_iter=1_000_000,
            tol=1e-4,
        )),
    ])

    if cv is not None:
        # cross-fitted mean on train (prevents overfit residuals)
        mean_train_cf = cross_val_predict(model, X_train, y_train, cv=cv, method="predict")
        mean_train_cf = pd.Series(mean_train_cf, index=y_train.index)
    else:
        # last-resort: in-sample fit (less ideal but stable)
        model.fit(X_train, y_train)
        mean_train_cf = pd.Series(model.predict(X_train), index=y_train.index)

    # fit final mean model on full train for test prediction
    model.fit(X_train, y_train)
    mean_test = float(model.predict(X_test)[0])

    y_resid_train = y_train - mean_train_cf
    return y_resid_train, mean_test, mean_train_cf