from pathlib import Path
import sys
import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
import yaml
import argparse
from scipy.stats import nct

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

# get path for utils
current_dir = Path(__file__).resolve().parent
scripts_root = current_dir.parent.parent
sys.path.insert(0, str(scripts_root))

# import needed utils
from Scripts.Utils.metrics import (
    calculate_crps,
    calculate_rmse,
    calculate_crps_quantile,
    shap_values,
)
from Scripts.Utils.density_fitting import fit_skew_t
from Scripts.Utils.qrf_utils import (
    get_pca,
    make_factor_features_time_safe,
)

# -----------------------
# Robust Linear Model Fitting (In-Sample)  [UNCHANGED]
# -----------------------
def fit_enet_mean_and_residuals(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    h: int,
    pub_lag: int = 2,
):
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    y_train = y_train.dropna()
    X_train = X_train.loc[y_train.index]

    n_samples = len(y_train)

    if n_samples < 30 or float(y_train.std()) == 0.0:
        return y_train.copy(), 0.0

    if n_samples < 50:
        n_splits = 2
    elif n_samples < 100:
        n_splits = 3
    else:
        n_splits = 5

    cv_gap = h + pub_lag
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=cv_gap)

    alphas = np.logspace(-3, 2, 50)
    l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95]

    pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("enet", ElasticNetCV(
            l1_ratio=l1_ratio,
            alphas=alphas,
            cv=tscv,
            n_jobs=-1,
            max_iter=100_000,
            selection='random',
            tol=1e-3,
        )),
    ])

    try:
        pipeline.fit(X_train, y_train)
    except Exception:
        return y_train.copy(), 0.0

    mean_train = pipeline.predict(X_train)
    mean_test = float(pipeline.predict(X_test)[0])
    y_resid_train = y_train - mean_train

    if y_resid_train.std() > y_train.std():
        return y_train.copy(), 0.0

    return pd.Series(y_resid_train, index=y_train.index), mean_test


# -----------------------
# Main Experiment
# -----------------------
def run_experiment(config):
    print(f"run {config['experiment_name']}")

    data_filename = config["data"].get("data_file", "data_stationary.csv")
    project_root = current_dir.parent.parent
    data_path = project_root / "Data" / "Cleaned_Data" / data_filename
    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)

    data_yoy_path = project_root / "Data" / "Cleaned_Data" / "data_yoy.csv"
    df_yoy = pd.read_csv(data_yoy_path, index_col="Date", parse_dates=True)

    targets = config["data"]["targets"]
    horizons = config["data"]["horizons"]
    eval_start_date = pd.Timestamp(config["data"]["eval_start_date"])
    snb_months = [3, 6, 9, 12]

    # Match Code 1 quantile grids
    plot_quantiles = [0.05, 0.16, 0.50, 0.84, 0.95]
    eval_quantiles = np.linspace(0.01, 0.99, 99)

    training_offset = 14
    pub_lag = 2

    use_lin_feat = bool(config.get("model", {}).get("use_linear_features", False))
    use_pca_factors = bool(config.get("model", {}).get("use_pca_factors", False))
    final_params = config["model"]["params"]

    for target_name in targets:
        for h in horizons:
            pca_bundle_fixed = None

            if target_name == "Headline":
                target_col = f"target_headline_{h}m"
                yoy_col = "Headline"
                yoy_raw = "Headline_level"
            else:
                target_col = f"target_core_{h}m"
                yoy_col = "Core"
                yoy_raw = "Core_level"

            if target_col not in df.columns:
                continue

            target_cols_to_drop = [c for c in df.columns if "target_" in c]

            requested_start_idx = df.index.get_loc(eval_start_date)
            if isinstance(requested_start_idx, slice):
                requested_start_idx = requested_start_idx.start
            start_idx = max(requested_start_idx, training_offset)
            total_rows = len(df)

            recursive_preds = []
            current_idx = start_idx

            while current_idx < total_rows:
                current_date = df.index[current_idx]
                forecast_date = current_date
                target_date = forecast_date + pd.DateOffset(months=h)

                if current_date.month not in snb_months:
                    current_idx += 1
                    continue

                last_trainable_idx = current_idx - h - pub_lag
                if last_trainable_idx < 0:
                    current_idx += 1
                    continue

                # --- Match Code 1 expanding window indices ---
                train_indices = range(training_offset, last_trainable_idx + 1)

                X_slice = df.drop(columns=target_cols_to_drop)
                Y_slice = df[target_col]

                X_train = X_slice.iloc[train_indices].copy()
                Y_train = Y_slice.iloc[train_indices].copy()
                X_test = X_slice.iloc[[current_idx]].copy()

                # Clean target and align X (Code 1 behavior)
                Y_train = Y_train.dropna()
                X_train = X_train.loc[Y_train.index]

                # PCA Generation (same structure as Code 1)
                if use_pca_factors:
                    pca_cols, keep_cols = get_pca(
                        df_columns=X_train.columns,
                        target_cols_to_drop=target_cols_to_drop,
                        target_name=target_name,
                        config=config,
                    )

                    if pca_bundle_fixed is None:
                        X_train, X_test, pca_bundle_fixed = make_factor_features_time_safe(
                            X_train=X_train,
                            X_test=X_test,
                            pca_cols=pca_cols,
                            keep_cols=keep_cols,
                            config=config,
                            forecast_date=forecast_date,
                            target_name=target_name,
                            h=h,
                            top_k=5,
                            pca_bundle=None,
                        )
                    else:
                        X_train, X_test, _ = make_factor_features_time_safe(
                            X_train=X_train,
                            X_test=X_test,
                            pca_cols=pca_bundle_fixed["pca_cols"],
                            keep_cols=pca_bundle_fixed["keep_cols"],
                            config=config,
                            forecast_date=forecast_date,
                            target_name=target_name,
                            h=h,
                            top_k=5,
                            pca_bundle=pca_bundle_fixed,
                        )

                # -------------------------------
                # Linear Features (KEEP Code 2 logic)
                # -------------------------------
                if use_lin_feat:
                    y_resid_train, mean_test = fit_enet_mean_and_residuals(
                        X_train=X_train,
                        y_train=Y_train,
                        X_test=X_test,
                        h=h,
                        pub_lag=pub_lag,
                    )
                    X_train_used = X_train.loc[y_resid_train.index].copy()
                    Y_train_used = y_resid_train.copy()

                    # Code 2's robust feature cleanup for ENet/QRF residual path
                    meds = X_train_used.median(numeric_only=True)
                    X_train_used = X_train_used.fillna(meds).fillna(0.0)
                    X_test_used = X_test.fillna(meds).fillna(0.0)

                    X_train_used = X_train_used.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                    X_test_used = X_test_used.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                else:
                    mean_test = 0.0
                    X_train_used = X_train
                    Y_train_used = Y_train
                    X_test_used = X_test

                # QRF Fit (default params + random_state)
                model_args = final_params.copy()
                model_args["random_state"] = 42
                model = RandomForestQuantileRegressor(**model_args)
                model.fit(X_train_used, Y_train_used)

                # Predict plot quantiles (match Code 1)
                preds_plot = model.predict(X_test_used, quantiles=list(plot_quantiles))
                # Predict dense grid (match Code 1)
                preds_dense = model.predict(X_test_used, quantiles=list(eval_quantiles))

                # If residual path, add mean back (Code 2 logic)
                if use_lin_feat:
                    preds_plot = preds_plot + mean_test
                    preds_dense = preds_dense + mean_test

                # -------------------------------
                # Direct Target Evaluation (match Code 1)
                # -------------------------------
                actual_direct = df.loc[forecast_date, target_col]
                if pd.isna(actual_direct):
                    current_idx += 1
                    continue

                y_fit_direct = preds_dense.flatten()
                skew_params_direct = fit_skew_t(y_fit_direct, eval_quantiles)

                crps_direct = calculate_crps(actual_direct, skew_params_direct)
                crps_direct_empirical = calculate_crps_quantile([actual_direct], preds_dense, eval_quantiles)

                pit_direct = nct.cdf(
                    actual_direct,
                    skew_params_direct[0],
                    skew_params_direct[1],
                    loc=skew_params_direct[2],
                    scale=skew_params_direct[3],
                )
                #get distribution parameters for logpdf calculation
                df_nct, nc_nct, loc_nct, scale_nct =skew_params_direct
                logpdf_direct=float(nct.logpdf(actual_direct, df_nct, nc_nct, loc=loc_nct, scale=scale_nct))
                #log predictive density (higher is better).
                logS_direct = logpdf_direct
                rmse_direct = calculate_rmse(actual_direct, preds_plot[0, 2])

                # SHAP (match Code 1 style)
                shap_tree = shap_values(model, X_test_used, X_train=X_train_used, model_type="tree")
                final_shap = {f"SHAP_{k}": float(np.asarray(v).squeeze()) for k, v in shap_tree.items()}

                # -------------------------------
                # YoY Reconstruction (match Code 1)
                # -------------------------------
                if target_date not in df_yoy.index:
                    current_idx += 1
                    continue

                T = target_date
                actual_yoy = df_yoy.loc[T, yoy_col]
                if pd.isna(actual_yoy):
                    current_idx += 1
                    continue

                scaling = h / 12.0
                if h == 12:
                    base_effect_expost = 0.0
                    preds_plot_yoy_expost = preds_plot.copy()
                    preds_dense_yoy_expost = preds_dense.copy()
                else:
                    lower = T - pd.DateOffset(months=12)
                    if (forecast_date not in df_yoy.index) or (lower not in df_yoy.index):
                        current_idx += 1
                        continue

                    p_t = np.log(df_yoy.loc[forecast_date, yoy_raw])
                    p_low = np.log(df_yoy.loc[lower, yoy_raw])
                    base_effect_expost = 100.0 * (p_t - p_low)

                    preds_plot_yoy_expost = base_effect_expost + preds_plot * scaling
                    preds_dense_yoy_expost = base_effect_expost + preds_dense * scaling

                # -------------------------------
                # Time-safe YoY (match Code 1)
                # -------------------------------
                t_known = forecast_date - pd.DateOffset(months=pub_lag)
                crps_yoy_timesafe_parametric = np.nan
                crps_yoy_timesafe_empirical = np.nan

                if h == 12:
                    preds_dense_yoy_timesafe = preds_dense.copy()
                else:
                    lower_known = T - pd.DateOffset(months=12)
                    if (t_known in df_yoy.index) and (lower_known in df_yoy.index):
                        p_known = np.log(df_yoy.loc[t_known, yoy_raw])
                        p_low = np.log(df_yoy.loc[lower_known, yoy_raw])
                        base_effect_timesafe = 100.0 * (p_known - p_low)
                        preds_dense_yoy_timesafe = base_effect_timesafe + preds_dense * scaling
                    else:
                        preds_dense_yoy_timesafe = None
                
                #initialize for quantile evaluation of time-safe yoy distribution
                q05_yoy_timesafe =np.nan
                q16_yoy_timesafe=np.nan
                q84_yoy_timesafe= np.nan
                q95_yoy_timesafe= np.nan
                median_yoy_timesafe=np.nan                
                violation_90_timesafe= np.nan       
                upper_violation_95_timesafe = np.nan  

                if preds_dense_yoy_timesafe is not None:
                    yoy_q = np.asarray(preds_dense_yoy_timesafe).reshape(1, -1)
                    crps_yoy_timesafe_empirical = float(
                        np.mean(calculate_crps_quantile([actual_yoy], yoy_q, eval_quantiles))
                    )
                    skew_params_yoy = fit_skew_t(yoy_q.flatten(), eval_quantiles)
                    crps_yoy_timesafe_parametric = float(calculate_crps(actual_yoy, skew_params_yoy))
                    #get quantiles 
                    q05_yoy_timesafe=float(np.percentile(preds_dense_yoy_timesafe, 5))
                    q16_yoy_timesafe= float(np.percentile(preds_dense_yoy_timesafe, 16))
                    q84_yoy_timesafe= float(np.percentile(preds_dense_yoy_timesafe, 84))
                    q95_yoy_timesafe= float(np.percentile(preds_dense_yoy_timesafe, 95))
                    median_yoy_timesafe=float(np.median(preds_dense_yoy_timesafe))
                    #Bool whether forecast falls outside 90% interval (from 5% and 95% quantiles)
                    violation_90_timesafe=int((actual_yoy <q05_yoy_timesafe) or (actual_yoy >q95_yoy_timesafe))
                    upper_violation_95_timesafe = int(actual_yoy> q95_yoy_timesafe)   #bool if actual > q95


                # Result dict (match Code 1 columns)
                result = {
                    "Date": forecast_date,
                    "Target_date": target_date,
                    "Actual_direct": actual_direct,
                    "Forecast_median_direct": preds_plot[0, 2],
                    "CRPS_direct_parametric": crps_direct,
                    "CRPS_direct_empirical": crps_direct_empirical,
                    "LogS_direct": logS_direct,
                    "RMSE_direct": rmse_direct,
                    "PIT_direct": pit_direct,
                    "df_skewt_direct": float(skew_params_direct[0]),
                    "nc_skewt_direct": float(skew_params_direct[1]),
                    "loc_skewt_direct": float(skew_params_direct[2]),
                    "scale_skewt_direct": float(skew_params_direct[3]),
                    "Actual_YoY": actual_yoy,
                    "Forecast_median_YoY": preds_plot_yoy_expost[0, 2],
                    "q05_YoY": preds_plot_yoy_expost[0, 0],
                    "q16_YoY": preds_plot_yoy_expost[0, 1],
                    "q84_YoY": preds_plot_yoy_expost[0, 3],
                    "q95_YoY": preds_plot_yoy_expost[0, 4],
                    "BaseEffect_YoY_expost": float(base_effect_expost),
                    "CRPS_YoY_timesafe_parametric": crps_yoy_timesafe_parametric,
                    "CRPS_YoY_timesafe_empirical": crps_yoy_timesafe_empirical,
                    'CRPS_YoY_timesafe_empirical': float(crps_yoy_timesafe_empirical) if np.isfinite(crps_yoy_timesafe_empirical) else np.nan, "q05_YoY_timesafe": q05_yoy_timesafe, "q16_YoY_timesafe": q16_yoy_timesafe, "q84_YoY_timesafe": q84_yoy_timesafe, "q95_YoY_timesafe": q95_yoy_timesafe,
                    "Violation90_YoY_timesafe": violation_90_timesafe, "UpperViolation95_YoY_timesafe": upper_violation_95_timesafe
                }
                result.update(final_shap)
                recursive_preds.append(result)

                current_idx += 1

                # Save (same cadence as your scripts)
                results_df = pd.DataFrame(recursive_preds).set_index("Date")
                save_name = Path(
                    f"Results/Data_experiments_qrf/{config['experiment_name']}_{target_name}_{h}m.csv"
                )
                save_name.parent.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        conf = yaml.safe_load(f)

    run_experiment(conf)