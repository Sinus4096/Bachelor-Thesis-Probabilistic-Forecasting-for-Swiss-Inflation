from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

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
def generate_linear_feature_oof(df, target_col, target_cols_to_drop, h, config, use_pca=False, window_size=120, min_train=40):
    """
    lin pred with walk forward approach, and pca if requestesd"""
    #initialize bundle for pca objects to be reused
    pca_bundle= None
    #drop the target related columns to get feature matrix
    X_full=df.drop(columns=target_cols_to_drop) 
    #isolate the target column for training
    y_full= df[target_col] 
    #initialize empty series for predictions with same index
    preds=pd.Series(index=df.index, dtype=float) 
    #ts splits def
    n_eff=len(y_full)
    #5 splits only if have enough data
    n_splits_eff =min(5, max(2, n_eff //15))  
    #set gap for cv to avoid using future info in short horizons
    cv_gap= 0 if h >= 6 else h   
    #setup time series split object for cross validation
    tscv =TimeSeriesSplit(n_splits=n_splits_eff, gap=cv_gap)
    
    #check if pca reduction is requested
    if use_pca: 
        #get columns to be reduced vs columns to keep as-is
        pca_cols, keep_cols= get_pca(df_columns=X_full.columns, target_cols_to_drop=[], target_name=None,  config=config)
        #only fit pca if it hasn't been initialized yet
        if use_pca and pca_bundle is None:
            #Fit PCA ONCE on full available sample (or better: first training window)
            scaler_full= StandardScaler()
            #determine end of the first training sample for pca fit
            initial_train_end= min_train
            #slice initial data to avoid look-ahead bias in factor construction
            X_initial= X_full.iloc[:initial_train_end]
            #fit and transform the pca subset
            Z_full= scaler_full.fit_transform(X_initial[pca_cols])
            #determine number of factors r based on explained variance (see config)
            r= choose_r_from_train_std(Z_full, config)
            #initialize pca model with r components
            pca_full= PCA(n_components=r)
            #fit the pca model on the standardized data
            pca_full.fit(Z_full)

            #sign stabilization (ensure first load is positive for interpretability)
            for j in range(pca_full.components_.shape[0]):
                if pca_full.components_[j, 0] < 0:
                    #flip sign of component if negative
                    pca_full.components_[j, :]*= -1

            #store objects in bundle to use in the walk-forward loop
            pca_bundle= {"scaler": scaler_full, "pca": pca_full, "r": r}

    #if no pca use all columns as keep columns
    else: 
        #no pca columns selected
        pca_cols=[] 
        #all features are treated as keep_cols
        keep_cols= list(X_full.columns)         
    #check if horizon is long term (for choosing grid in config)
    if h>= 12: 
        #use higher alpha grid for long horizons (more regularization)
        enet= ElasticNetCV(l1_ratio=[0.01, 0.03, 0.05, 0.10, 0.20], alphas=np.logspace(-3, 2.5, 60), cv=tscv, n_jobs=-1,max_iter=1_000_000, tol=1e-4)
    #else use short term grid
    else: 
        #standard alpha grid for short horizons
        enet= ElasticNetCV(l1_ratio=[0.05, 0.10, 0.20, 0.35, 0.50], alphas=[0.1,0.5, 1.0,25, 10.0, 50.0], cv=tscv, n_jobs=-1, max_iter=1_000_000, tol=1e-4)
    
    #initialize the standard scaler for feature normalization
    scaler= StandardScaler() 
    #loop through every time step in the dataframe: walk fwd approach so no data leakage
    for i in range(len(df)):         
        #define the end of the training window based on horizon h
        train_end_idx=i-h         
        #ensure we have enough data to train
        if train_end_idx < min_train: 
            #set prediction to zero if data is insufficient (early observations)
            preds.iloc[i]= 0.0 
            #skip to next iteration
            continue 
        #calculate start index for rolling window (based on window_size in config)
        train_start_idx=max(0, train_end_idx -window_size)        
        #slice feature matrix for training window
        X_train_raw=X_full.iloc[train_start_idx : train_end_idx +1] 
        #slice target vector for  training window
        y_train= y_full.iloc[train_start_idx : train_end_idx +1] 
        #slice current feature row for testing 
        X_test_raw= X_full.iloc[[i]]         
        #identify non-NaN values in the target
        valid_mask= ~y_train.isna() 
        #remove NaNs from training target
        y_train_clean=y_train[valid_mask] 
        #remove corresponding rows from training features
        X_train_clean= X_train_raw[valid_mask]         
        #validate if enough samples remain for CV and check variance
        if len(y_train_clean) <(n_splits_eff +2) or y_train_clean.std()== 0: 
            #default to zero if training is impossible (no variance)
            preds.iloc[i]= 0.0 
            #skip to next iteration
            continue 

        #check for pca path to transform features into factors
        if use_pca: 
            #handle columns that skip pca (e.g. dummies or specific indicators)
            if len(keep_cols) > 0: 
                #extract values for keep columns in train
                X_train_keep= X_train_clean[keep_cols].values 
                #extract values for keep columns in test
                X_test_keep=X_test_raw[keep_cols].values 
            #else initialize empty arrays for stacking logic
            else: 
                #create empty train array for hstack
                X_train_keep= np.empty((len(X_train_clean), 0)) 
                #create empty test array for hstack
                X_test_keep= np.empty((len(X_test_raw), 0)) 

            #handle columns requiring pca (factor estimation)
            if len(pca_cols) > 0: 
                #get scaler from bundle to ensure consistent transformation
                scaler_full= pca_bundle["scaler"]
                #get fitted pca object from bundle
                pca_full= pca_bundle["pca"]
                #standardize train data based on pre-fitted scaler
                X_train_std= scaler_full.transform(X_train_clean[pca_cols])
                #project train data onto principal components
                F_train= pca_full.transform(X_train_std)
                #standardize test data
                X_test_std= scaler_full.transform(X_test_raw[pca_cols])
                #project test data onto factors
                F_test= pca_full.transform(X_test_std)
                #horizontally stack keep features and factors for train
                X_train_final= np.hstack([X_train_keep, F_train]) 
                X_test_final= np.hstack([X_test_keep, F_test]) 
            #if no pca columns just use keep columns for final matrix
            else: 
                #final train is just keep
                X_train_final= X_train_keep 
                #final test is just keep
                X_test_final= X_test_keep 

        #path for raw feature scaling (if pca is disabled)
        else: 
            #fit and transform train features with standard scaling
            X_train_final= scaler.fit_transform(X_train_clean) 
            #transform test features using train scaling parameters (avoiding leakage)
            X_test_final= scaler.transform(X_test_raw) 

        #create a fresh copy of the model for this window (reset weights)
        m=clone(enet) 
        #fit model on processed training data (factors or raw features)
        m.fit(X_train_final, y_train_clean) 
        #predict current step and store in series for later evaluation
        preds.iloc[i]= m.predict(X_test_final)[0] 

    #fill any remaining missing values with zero (stability check)
    preds= preds.fillna(0.0) 
    #return the full series of predictions for the current target/horizon
    return preds


