import numpy as np
from scipy import stats
import pandas as pd
import numpy as np

def crps_from_samples(y, samples):
    """
    proper crps for an empirical predictive distribution
    """
    #convert samples to flat array for calculation
    s= np.asarray(samples).ravel()
    #return nan if no samples or target is missing
    if s.size==0 or np.isnan(y):
        return np.nan
    #calc absolute error term
    term1= np.mean(np.abs(s -y))
    #calc internal sample distance term
    term2= 0.5*np.mean(np.abs(s[:, None]-s[None, :]))
    #return final crps score
    return float(term1 - term2)



def rolling_crps_score(data: pd.DataFrame, target_col: str| None= None, target_idx: int= 0, horizon: int= 12, prior_type: str= "natural_niw",
    prior_params: dict | None= None, fixed_lambda: float= 0.02, start_eval: int= 60,
    step: int= 1, n_draws: int= 600, burn_in: int= 150):
    """
    rolling crps """

    #detect target column 
    if target_col is None:
        #look for columns with target prefix
        target_cols= [c for c in data.columns if "target_" in c]
        if not target_cols:
            #raise error if no targets identified
            raise ValueError("no target column found (expected columns containing 'target_').")
        #default to first target found
        target_col= target_cols[0]
    if target_col not in data.columns:
        #validate column exists in provided dataframe
        raise ValueError(f"target_col='{target_col}' not in data.columns")
    #get integer location for faster access
    target_col_loc= data.columns.get_loc(target_col)
    #get total sample size
    T= len(data)
    #initialize list for results
    scores: list[float]= []
    #ensure indices exist for history and training
    min_t= max(start_eval, horizon + 1)

    #loop through time origins for evaluation
    for t in range(min_t, T, step):
        #truth for origin t is stored at row t (pre-shifted)
        y_true= data.iloc[t, target_col_loc]
        #skip if realization is missing
        if pd.isna(y_true):
            continue
        #training ends at t-h to prevent leakage of future info
        train_end_idx= t-horizon
        #check for minimum training observations
        if train_end_idx < 15:
            continue

        #slice training data up to valid end point
        df_train= data.iloc[: train_end_idx + 1].copy()

        #forecast info up to current origin t
        df_forecast= data.iloc[: t + 1].copy()
        #overwrite last row with actual known target value y_t
        df_forecast.iloc[-1, target_col_loc]= data.iloc[t -horizon, target_col_loc]
        #overwrite previous row with known target y_{t-1}
        df_forecast.iloc[-2, target_col_loc]= data.iloc[t -horizon-1, target_col_loc]
        #fit model and generate forecast
        b= BVAR(lags=2, prior_type=prior_type, prior_params=prior_params)         #initialize bvar with specific prior setup
        b.fit(df_train, horizon=horizon, n_draws=n_draws, burn_in=burn_in, fixed_lambda=fixed_lambda) #fit on isolated training slice

        #generate predictive distribution for current origin
        all_preds= b.forecast(df_forecast)
        #isolate predictive draws for target variable
        draws= np.asarray(all_preds)[:, target_idx]
        #calc score for this time step
        score= crps_from_samples(y_true, draws)
        #store valid scores
        if not np.isnan(score):
            scores.append(float(score))

    #return mean crps across all evaluated origins
    return float(np.mean(scores)) if scores else np.nan


#see thesis for formulas
class BVAR:
    """
    implementation of Bayesian VAR with natural conjugate priors
    """
    def __init__(self, lags=2, prior_type='minnesota',prior_params=None, implementation_type= 'dummies'):
        #initialize model
        self.p= lags  #number of lags
        self.prior_type =prior_type      #is minnesota by default
        self.implementation_type= implementation_type  #dummies or analytical (to see which more stable)
        self.params= prior_params if prior_params else {'lambda': 0.2, 'theta':0.5, 'a3': 100.0, 'alpha':2.0} #default prior params if none provided
        self.phi_draws= None        #to store posterior draws of coefficients
        self.sigma_draws= []  #to store posterior draws of error variances

        self.n_vars= None      #to store number of variables
        self.n_features= None  #to store number of features after lags (K= N*p +1)
        self.feature_var_indices =None #to track which variable owns a feature column
        #initialize cache for tuning 
        self._mn_tuned_cache ={}
        self._ind_tuned_cache ={}


    def create_lags(self, data, horizon =12):
        #need to know horizon to avoid data leakage
        self.h=horizon
        #get target and feature columns
        target_cols= [c for c in data.columns if 'target_' in c]
        feature_cols= [c for c in data.columns if 'target_' not in c]
        #extract raw values
        Y_raw=data[target_cols].values
        X_raw= data[feature_cols].values
        #get number of variables and observations
        self.n_vars = Y_raw.shape[1]
        self.n_exog= X_raw.shape[1] #nr exogenous features
        n_obs= len(data)
        
        #define want lags 0, 1
        pub_lag =2
        #use information available at origin: t-2 and t-3
        self.lag_indices = [pub_lag, pub_lag+1 ]

        # must cover BOTH horizon shift and pub lag
        max_lag = self.h+max(self.lag_indices)    # h + 3
        #if less obs than max lag +1 -> cannot create lagged features-> raise error
        if n_obs <= max_lag:
            raise ValueError(f"Data has {n_obs} rows, but Lag 12 requires at least 13 observations.")
        #initialize list to store lagged features
        X_list= []
        #identify which cols are targets vs features for prior
        feature_type_map1 =[] 
        #loop through lags and create lagged features
        for lag in self.lag_indices:
            start= max_lag -lag  #start point for this lag
            end =n_obs- lag     #end point for this lag
            #for target lags at time t-lag, need to shift back by h
            target_start =max_lag-(lag+self.h)
            target_end =n_obs-(lag+self.h)
            X_list.append(Y_raw[target_start:target_end, :])  #append lagged targets to list
            feature_type_map1.extend([0]* self.n_vars)  #mark lagged targets as type 0 for prior purposes
            X_list.append(X_raw[start:end, :])     #append lagged features
            feature_type_map1.extend([1] * self.n_exog)
        #combine lagged features
        X_combined=np.column_stack(X_list)
        #want to deduplicate if lags of different variables are the same
        df_temp =pd.DataFrame(X_combined) 
        is_duplicate= df_temp.T.duplicated().values  #boolean mask for duplicate columns
        self.kept_indices= np.where(~is_duplicate)[0]   #store indices of unique features
        #determine which variable index (0 to N-1) each feature belongs to
        block_size=self.n_vars+self.n_exog
        self.feature_var_indices= []        
        for k in self.kept_indices:
            #k is the index in the original huge stack
            #mod block_size gives the variable index
            original_var_idx = k % block_size
            self.feature_var_indices.append(original_var_idx)
        self.feature_var_indices = np.array(self.feature_var_indices)
        
        #keep only unique features
        X_unique =X_combined[:, self.kept_indices]
        #keep map aligned with unique columns
        self.feature_type_map = np.array(feature_type_map1)[self.kept_indices]
        #add intercept based error
        X =np.column_stack([np.ones(X_unique.shape[0]), X_unique])
        #align Y: Y starts from max_lag to match the lag history
        Y_aligned =Y_raw[max_lag:, :]        
        self.n_features =X.shape[1]  #number of features after adding intercept and deduplication
        return X, Y_aligned

 
    def natural_moments(self, N, lam, a3, sigmas_y, sigmas_x, theta, alpha_decay):
        """ constructs prior moments for natural conjugate normal-inverse wishart prior"""
        #get dimensions
        K= self.n_features
        #manually control how much we trust exogenous vs target lags
        theta = float(theta)
        alpha = float(alpha_decay)    
        #prior mean: identity for first own-lag, zeros otherwise
        Phi_0= np.zeros((K, N))         
        #prior tightness matrix
        psi_diag =np.zeros(K)        
        #intercept tightness
        psi_diag[0]= a3  
        #get cols per lag block 
        block_size = self.n_vars+self.n_exog
        #loop through lag indices to apply shrinkage
        for j, f_type in enumerate(self.feature_type_map):
            #column index in the full Phi matrix (including intercept->+1)
            col_idx= j+1             
            #index in X_combinded
            orig_j =int(self.kept_indices[j])
            #use the specific sigma for this feature
            s_jj= sigmas_x[j]    
            #lag distance
            lag_pos =orig_j//block_size
            lag=self.lag_indices[lag_pos]     
            # Find if this column is target (0) or exogenous (1)
            f_type = self.feature_type_map[j]            
            if f_type== 0:
                #target lags (Standard Minnesota)
                lag_dist = self.h + max(1, lag)  # y_{t-(h+lag)}
                psi_diag[col_idx] = (lam**2) / (s_jj**2 * (lag_dist**alpha))
            else:
                #exogenous features (Extra Tightness to handle multicollinearity)
                lag_dist=max(1, lag)           # x_{t-lag}
                psi_diag[col_idx]= (theta *lam**2)/(s_jj**2*(lag_dist**alpha))
        return Phi_0, np.diag(psi_diag)


    def fit(self, data, horizon=12, n_draws=2000, burn_in=500, fixed_lambda=None):
        """estimate bvar
        """
        #create lagged matrices calling fct
        X, Y= self.create_lags(data, horizon=horizon)
        #get dimensions
        T, N= Y.shape    #nr of equations and obs.
        K= self.n_features  #nr of features

        #calc res variances from univariate ar for targets
        sigmas= []  #to store scales
        for idx in range(N):
            #fit univariate ar to get residual std
            res= np.diff(Y[:, idx])
            sigmas.append(np.std(res) if len(res)>0 else 0.1)
        sigmas= np.array(sigmas)

        #extract feature columns before they were lagged
        feature_cols= [c for c in data.columns if 'target_' not in c]
        #get raw feature data (without lags) 
        X_raw_data= data[feature_cols].values  
        #remove intercept for scaling calc
        feature_cols_raw= X[:, 1:] 
        sigmas_x_all= []
        for idx in range(feature_cols_raw.shape[1]):
            #calc variance for predictors
            res= np.diff(feature_cols_raw[:, idx])
            sigmas_x_all.append(np.std(res) if len(res) > 0 else 0.1)
        sigmas_all= np.array(sigmas_x_all)

        #minnesota config
        #-----------------
        if 'minnesota' in self.prior_type:
            #intercept prior variance
            a3= float(self.params.get('alpha', 100.0))  
            #use instance-level cache for tuning
            cache_key= ("minnesota", horizon, int(self.n_features), int(self.n_vars))

            if fixed_lambda is not None:
                #use provided lambda
                best_lam= float(fixed_lambda)
                best_theta= float(self.params.get('theta', 0.01))
                best_decay= float(self.params.get('alpha_decay', 2.0))
            else:
                #check if results are already in cache
                if cache_key in self._mn_tuned_cache:
                    best_lam, best_theta, best_decay= self._mn_tuned_cache[cache_key]
                else:
                    #best lambda=a1=a2 (match natural niw properties)
                    if horizon <= 3:
                        lambda_grid = [0.01, 0.02, 0.05, 0.07, 0.1]
                        decay_grid  = [1.0, 1.25, 1.5]
                        theta_grid  = [0.01, 0.03, 0.05, 0.1]

                    elif horizon <= 6:
                        lambda_grid = [0.02, 0.05, 0.07, 0.1, 0.15, 0.2]
                        decay_grid  = [1.0, 1.25, 1.5, 2.0]
                        theta_grid  = [0.03, 0.05, 0.1, 0.2, 0.3]

                    elif horizon <= 9:
                        lambda_grid = [0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
                        decay_grid  = [1.25, 1.5, 2.0]
                        theta_grid  = [0.05, 0.1, 0.2, 0.3, 0.5]

                    else:  # horizon >= 12
                        lambda_grid = [0.07, 0.1, 0.15, 0.2, 0.3, 0.4]
                        decay_grid  = [1.25, 1.5, 2.0]
                        theta_grid  = [0.1, 0.2, 0.3, 0.5, 0.7] 

                    #cheaper tuning settings for performance
                    start_eval= max(horizon + 24, int(0.75 * len(data)))
                    step_tune= 4
                    n_draws_tune= 150
                    burn_in_tune= 50
                    #get target variable for crps evaluation
                    target_cols= [c for c in data.columns if "target_" in c]
                    target_col= target_cols[0]
                    #initialize best parameters
                    best_score= np.inf
                    best_params_tuple= (0.2, 0.5, 2.0)
                    #iterate through grid
                    for lam in lambda_grid:
                        for theta in theta_grid:
                            for dec in decay_grid:
                                #config params for current step
                                p_params= {"theta": float(theta), "alpha_decay": float(dec), "alpha": float(a3)}
                                #call rolling evaluation (with fixed_lambda to stop recursion)
                                score= rolling_crps_score(data=data, target_col=target_col,
                                    target_idx=0, horizon=horizon, prior_type="minnesota", prior_params=p_params,
                                    fixed_lambda=float(lam), start_eval=start_eval, step=step_tune, n_draws=n_draws_tune, burn_in=burn_in_tune)
                                #update best params if score is better
                                if score < best_score:
                                    best_score= score
                                    best_params_tuple= (float(lam), float(theta), float(dec))
                    #assign best params
                    best_lam, best_theta, best_decay= best_params_tuple                    
                    #store results in cache
                    self._mn_tuned_cache[cache_key]= (best_lam, best_theta, best_decay)
                    self.params.update({'lambda': best_lam, 'theta': best_theta, 'alpha_decay': best_decay})

            #estimation (equation-by-equation analytical ridge)
            Phi_post_all= np.zeros((K, N))
            V_post_list= []
            #calc data precision
            XX= X.T @X            
            block_size= self.n_vars+self.n_exog
            #standard minnesota does not shift by h
            use_h= False 
            for i in range(N):
                #construct diagonal penalty matrix p for equation i
                P_diag= np.zeros(K)
                #intercept precision
                P_diag[0]= 1.0 /a3 
                
                # --- CHANGE 1: Define Prior Mean Vector (Shrink to RW) ---
                # Initialize prior mean to 0
                Phi_prior_i = np.zeros(K) 
                
                for feat_j, f_type in enumerate(self.feature_type_map):
                    #offset for intercept
                    col_idx= feat_j +1 
                    #identify original indices and lag info
                    orig_k= int(self.kept_indices[feat_j])
                    lag_num= self.lag_indices[orig_k //block_size]
                    var_idx= self.feature_var_indices[feat_j]
                    
                    # --- CHANGE 2: Set Own First Lag to 1.0 ---
                    # If this feature is the same variable as the target (var_idx == i)
                    # AND it is the first lag (lag_num == 1)
                    if f_type == 0 and var_idx == i and lag_num == 1:
                        Phi_prior_i[col_idx] = 1.0

                    #sigma of predictor
                    sigma_j= sigmas_all[feat_j] 
                    #sigma of dependent variable
                    sigma_i= sigmas[i]          

                    #lag decay function logic
                    if f_type == 0:
                        #lag distance for endogenous
                        lag_dist= (self.h + max(1, lag_num)) if use_h else max(1, lag_num)
                    else:
                        #lag distance for exog
                        lag_dist= max(1, lag_num)
                    #calc lag influence based on tuned decay
                    lag_func= (lag_dist ** best_decay)
                    #ridge penalty calc
                    if f_type == 0 and var_idx ==i:
                        #own lag variance logic
                        ridge_penalty= (sigma_i**2) *((lag_func/best_lam)**2)
                    else:
                        #cross lag or exogenous variance logic
                        ridge_penalty= ((sigma_j *lag_func) /(best_lam*best_theta))**2

                    P_diag[col_idx]= ridge_penalty
                
                #posterior precision
                Post_Precision_i= XX + np.diag(P_diag) + np.eye(K) * 1e-6

                # --- CHANGE 3: Apply Prior Mean to Solver ---
                # Calculate the Right Hand Side: X'Y + P*Beta_prior
                # This pulls the coefficients towards Phi_prior_i instead of 0
                RHS = (X.T @ Y[:, i]) + (P_diag * Phi_prior_i)

                #solve for beta using cholesky for speed/stability
                L_i= np.linalg.cholesky(Post_Precision_i)
                w= np.linalg.solve(L_i, RHS) # Use the new RHS
                phi_i= np.linalg.solve(L_i.T, w)
                
                #store coefficients
                Phi_post_all[:, i]= phi_i
                #covariance for sampling
                L_inv= np.linalg.solve(L_i, np.eye(K))
                Unscaled_Cov= L_inv.T @L_inv
                V_post_list.append(Unscaled_Cov)
            
            # Calculate fitted values
            Y_hat = X @ Phi_post_all
            # Calculate actual residuals (in-sample error)
            Residuals = Y - Y_hat
            
            # Calculate standard deviation of realized residuals
            # This captures the actual h-step model uncertainty
            sigmas_posterior = np.std(Residuals, axis=0)
            
            # Ensure no zeros (sanity check)
            sigmas_posterior = np.maximum(sigmas_posterior, 1e-6)

            # --- 4. SAMPLING USING POSTERIOR SIGMAS ---
            # Use the FITTED sigmas for the forecast noise
            Sigma_fixed= np.diag(sigmas_posterior**2)             
            self.sigma_draws= np.repeat(Sigma_fixed[None, :, :], n_draws, axis=0) 

            self.phi_draws= np.empty((n_draws, K, N))
            for i in range(N):
                # --- CRITICAL: Use sigmas_posterior (residuals) not sigmas (prior proxy) ---
                # This ensures the sampling width matches the residuals you calculated above
                cov_i=(sigmas_posterior[i]**2) *V_post_list[i]
                
                #numerical jitter to ensure pd
                cov_i= cov_i +np.eye(K) *1e-10
                #draw coefficients via cholesky decomposition
                L= np.linalg.cholesky(cov_i)
                Z= np.random.standard_normal((n_draws, K))
                #apply draw to phi draws storage
                self.phi_draws[:, :, i]= Phi_post_all[:, i][None, :]+Z@ L.T
                
        #independent normal-inverse wishart prior
        #----------------------------------
        elif 'independent_niw' in self.prior_type:
            #def hyperpar
            a3= float(self.params.get('alpha', 100.0))    #intercept variance scale
            a2_default= float(self.params.get('theta', 0.01)) 
            #nr of iters and burn in 
            n_iter= int(self.params.get('sampling', {}).get('n_draws', n_draws))
            burn_in_local= int(self.params.get('sampling', {}).get('burn_in', burn_in))

            #dont want rolling window to aboid recursion in rolling crps      
            if fixed_lambda is not None:
                #use fixed hyperparameter if provided
                a1= float(fixed_lambda)
                #use default theta
                a2= a2_default          
                #set decay for lag influence
                decay= float(self.params.get("alpha_decay", 2.0))
            else:  #grid search
                #create cache key for tuned params
                cache_key= ("independent_niw", horizon, int(self.n_features), int(self.n_vars), int(self.n_exog))
                #check if params already in cache
                if cache_key in self._ind_tuned_cache:
                    #retrieve from cache
                    a1, a2, decay= self._ind_tuned_cache[cache_key]
                else:
                    #get k and exog vars to adjust grid depending on features
                    K= self.n_features
                    exog= self.n_exog
                    #set grids based on forecast horizon
                    if horizon <= 3:
                        # very short horizon → allow flexibility but avoid ultra-weak shrinkage
                        lambda_grid = [0.03, 0.05, 0.07, 0.1]
                        theta_grid  = [0.03, 0.05, 0.1]
                        decay_grid  = [1.0, 1.25, 1.5]

                    elif horizon <= 6:

                        lambda_grid = [0.05, 0.07, 0.1, 0.15, 0.2]  # drop 0.01/0.03 entirely
                        theta_grid  = [0.05, 0.1, 0.2, 0.3]
                        decay_grid  = [ 1.25]

                    elif horizon <= 9:
                        # multi-step → stronger shrinkage + stronger variance prior
                        lambda_grid = [0.07, 0.1, 0.15, 0.2, 0.3]
                        theta_grid  = [0.1, 0.2, 0.3, 0.5]
                        decay_grid  = [1.0, 1.25, 1.5]

                    else:  # horizon >= 12
                        # long horizon → stability dominates
                        lambda_grid= [0.05, 0.07, 0.09, 0.1, 0.2, 0.3] 
                        theta_grid= [0.05, 0.1, 0.3, 0.5] 
                        decay_grid= [1.0, 1.5, 2.0]
                    #set start for evaluation window
                    start_eval= max(horizon + 24, int(0.6*len(data)))
                    #step size for tuning
                    step_tune= 4
                    #small draws for speed in tuning
                    n_draws_tune= 150
                    burn_in_tune= 50

                    #isolate target columns for score calc
                    target_cols= [c for c in data.columns if "target_" in c]
                    target_col= target_cols[0]

                    #initialize best score tracker
                    best_score= np.inf
                    best_params_tuple= (lambda_grid[0], theta_grid[0], decay_grid[0])

                    #nested loops for hyperparameter grid search
                    for a1_loop in lambda_grid:
                        for th in theta_grid:
                            for dec in decay_grid:
                                #set seed for reproducibility in tuning
                                np.random.seed(123)
                                #setup param dict for current iteration
                                p_params= {"theta": float(th), "alpha_decay": float(dec), "alpha": float(a3), "sampling": {"n_draws": n_draws_tune, "burn_in": burn_in_tune}}

                                #calc crps score for current grid point
                                score= rolling_crps_score(data=data, target_col=target_col, target_idx=0, horizon=horizon, prior_type="independent_niw",
                                    prior_params=p_params, fixed_lambda=float(a1_loop), start_eval=start_eval, step=step_tune, n_draws=n_draws_tune,
                                    burn_in=burn_in_tune)
                                #update best params if score improves
                                if score < best_score:
                                    best_score= score
                                    best_params_tuple= (float(a1_loop), float(th), float(dec))

                    #assign tuned params
                    a1, a2, decay= best_params_tuple
                    #store results in cache
                    self._ind_tuned_cache[cache_key]= (a1, a2, decay)
                    #update model params
                    self.params.update({"lambda": a1, "theta": a2, "alpha_decay": decay})
            
            #prior moments
            alpha_0= np.zeros(N*K)  #all zeros for wn
            V_alpha_0_diag= np.zeros(N*K) #to store prior precision diag
            block_size= self.n_vars + self.n_exog #original block size before deduplication
            
            #loop through each equation and feature to set prior variances
            Phi_0= np.zeros((K, N))
            #determine which feature is 1st lag of dependent variable
            for n in range(N):
                #search for own-lag index in feature mapping
                for k_search in range(1, K):
                    feat_idx= k_search-1
                    #match variable and feature type
                    if self.feature_type_map[feat_idx] ==0 and self.feature_var_indices[feat_idx] ==n:
                        #find smallest lag
                        orig_k= int(self.kept_indices[feat_idx])
                        #check if lag 1
                        if self.lag_indices[orig_k //block_size] == 1: 
                            #set rw prior mean to 1 for own lag
                            Phi_0[k_search, n]= 1.0 
                            break
            
            #flatten prior mean matrix
            alpha_0= Phi_0.flatten(order='F')
            #corrected prior variance scaling
            for n in range(N): 
                for k in range(K):
                    idx= n*K+k
                    sigma_i= sigmas[n]
                    
                    #handle intercept separately
                    if k == 0: 
                        #set variance for intercept
                        V_alpha_0_diag[idx]= (sigma_i**2) *a3
                    else:
                        #lag/factor logic: define variables here
                        feat_j= k -1
                        f_type= self.feature_type_map[feat_j]
                        orig_k= int(self.kept_indices[feat_j])
                        #get lag number for decay calc
                        lag_num= self.lag_indices[orig_k // block_size]
                        var_idx= self.feature_var_indices[feat_j]
                        
                        #get standard deviation of feature
                        sigma_j= sigmas_all[feat_j]
                        #calc lag decay function
                        lag_func= (max(1, lag_num)** decay)

                        #perform minnesota check for variance scaling
                        if f_type ==0 and var_idx== n:
                            #set variance for own lags
                            V_alpha_0_diag[idx]= (a1 / lag_func)**2
                        elif f_type ==0 and var_idx!= n:
                            #set variance for cross lags
                            V_alpha_0_diag[idx]= ((a1 *a2*sigma_i)/(sigma_j*lag_func))**2
                        else:
                            #pca factors (exogenous block)
                            V_alpha_0_diag[idx]= ((a1 *a2* sigma_i) /(sigma_j * lag_func))**2
            
            #define prior precision as inverse of variance and make sure never hits 0
            V_alpha_0_diag= np.maximum(V_alpha_0_diag, 1e-12)
            V_alpha_0_inv= np.diag(1.0 / V_alpha_0_diag)
            
            #prior scale matrix for sigma
            S_0= np.diag(sigmas**2)
            #prior degrees of freedom
            nu_0= N+2
            #initialize for gibbs sampling
            Sigma_current= np.diag(sigmas**2)
            Sigma_current= np.atleast_2d(Sigma_current)  
            
            #initialize storage for draws
            keep= max(1, n_iter - burn_in_local)
            self.phi_draws= np.zeros((keep, K, N))
            self.sigma_draws= np.zeros((keep, N, N))
            #x'x for precision update
            XX= X.T @ X

            #iterate through draws to sample from posterior: gibbs sampling
            for it in range(n_iter):
                #take inverse of current sigma draw
                Sigma_inv= np.linalg.inv(Sigma_current)   
                #capture independent flexibility (posterior precision)
                V_alpha_post_inv= V_alpha_0_inv + np.kron(Sigma_inv, XX)
                #add tiny bit of jitter to ensure it is positive definite
                V_alpha_post_inv+= np.eye(V_alpha_post_inv.shape[0]) * 1e-9
                #use cholesky on precision matrix for stability
                L_upper= np.linalg.cholesky(V_alpha_post_inv).T 
                #calc weighted avg of prior and data
                data_term= (X.T @ Y @ Sigma_inv).flatten(order='F')
                rhs= V_alpha_0_inv @ alpha_0 + data_term
                #solve for posterior mean using cholesky factors
                alpha_hat= np.linalg.solve(L_upper, np.linalg.solve(L_upper.T, rhs))
                #draw alpha using standard normal z
                Z= np.random.standard_normal(N * K)
                #shift and scale z to get posterior draw
                alpha_draw= alpha_hat + np.linalg.solve(L_upper, Z)

                #reshape back to matrix form
                Phi_current= alpha_draw.reshape((K, N), order='F')

                #compute residuals for current coefficients
                residuals= Y -X@ Phi_current
                #calc posterior scale matrix for sigma
                S_post= S_0 +residuals.T@ residuals
                #posterior degrees of freedom
                nu_post= nu_0+T   
                    
                #draw new sigma from inverse-wishart
                Sigma_current= stats.invwishart.rvs(df=nu_post, scale=S_post)
                Sigma_current= np.atleast_2d(Sigma_current)  

                #store draws after burn-in period
                if it >= burn_in_local:
                    j= it -burn_in_local
                    if j < keep:
                        #save draw for coefficients
                        self.phi_draws[j]= Phi_current
                        #save draw for covariance
                        self.sigma_draws[j]= Sigma_current
                    

        #natural conjugate normal-wishart prior
        #-------------------------------------
        elif 'natural_niw' in self.prior_type:
            #hyperparams
            a3= float(self.params.get('alpha', 100.0))
            #use instance-level cache for tuning (like minnesota)
            cache_key = ("natural_niw", horizon, int(self.n_features), int(self.n_vars))
            #choose/tune lambda/theta/decay 
            if fixed_lambda is not None:
                #use provided lambda
                best_lam= float(fixed_lambda)
                best_theta= float(self.params.get('theta', 0.01))
                best_decay= float(self.params.get('alpha_decay', 2.0))
            else:
                # reuse cached tuning if available
                if hasattr(self, "_niw_tuned_cache") and cache_key in self._niw_tuned_cache:
                    best_lam, best_theta, best_decay = self._niw_tuned_cache[cache_key]
                else:
                    # init cache if missing
                    if not hasattr(self, "_niw_tuned_cache"):
                        self._niw_tuned_cache = {}

                    # horizon-aware grids (same philosophy as your improved minnesota ones)
                    if horizon <= 3:
                        lambda_grid = [0.01, 0.02, 0.03]
                        theta_grid  = [0.01, 0.03, 0.05]
                        decay_grid  = [1.5, 2.0]
                    else:
                        lambda_grid = [0.02, 0.03]
                        theta_grid  = [0.01, 0.03]
                        decay_grid  = [2.0, 2.5]


                    # tuning settings (same as minnesota)
                    start_eval   = max(horizon + 24, int(0.75 * len(data)))
                    step_tune    = 4
                    n_draws_tune = 150
                    burn_in_tune = 50

                    # target variable for CRPS evaluation
                    target_cols = [c for c in data.columns if "target_" in c]
                    target_col  = target_cols[0]

                    best_score = np.inf
                    best_params_tuple = (lambda_grid[0], theta_grid[0], decay_grid[0])

                    for lam in lambda_grid:
                        for theta in theta_grid:
                            for dec in decay_grid:
                                p_params = {"theta": float(theta), "alpha_decay": float(dec), "alpha": float(a3)}

                                score = rolling_crps_score(
                                    data=data,
                                    target_col=target_col,
                                    target_idx=0,
                                    horizon=horizon,
                                    prior_type="natural_niw",
                                    prior_params=p_params,
                                    fixed_lambda=float(lam),     # IMPORTANT to avoid recursion
                                    start_eval=start_eval,
                                    step=step_tune,
                                    n_draws=n_draws_tune,
                                    burn_in=burn_in_tune
                                )

                                if score < best_score:
                                    best_score = score
                                    best_params_tuple = (float(lam), float(theta), float(dec))

                    best_lam, best_theta, best_decay = best_params_tuple
                    self._niw_tuned_cache[cache_key] = (best_lam, best_theta, best_decay)

            #persist chosen params so forecast/eval sees them
            self.params['lambda']= best_lam
            self.params['theta']= best_theta
            self.params['alpha_decay']= best_decay

            #build sigmas for predictors& remove intercept to calc variance of features only
            X_no_intercept= X[:, 1:]
            sigmas_x_all= []
            for j in range(X_no_intercept.shape[1]):
                #calc first differences for predictor scale
                r= np.diff(X_no_intercept[:, j])
                #avoid zero variance issues
                sigmas_x_all.append(np.std(r) if r.size > 1 else 0.1)
            sigmas_x_all= np.array(sigmas_x_all)
            #get minnesota-style prior moments for conjugate setup
            Phi_0, Psi_0= self.natural_moments(N, best_lam, a3, sigmas, sigmas_x_all, best_theta, best_decay)

            #standard mniw posterior 
            nu_0= N + 2
            #diagonal prior scale matrix
            S_0= np.diag(sigmas**2)
            #ensure matrix format for precision calc
            Psi_0= np.atleast_2d(Psi_0)
            Psi_0_inv= np.linalg.inv(Psi_0)
            #calc data cross-products
            XX= X.T @X
            XY= X.T@ Y
            #posterior right scale matrix for phi
            Psi_post_inv= Psi_0_inv+XX
            #add jitter for numerical stability
            Psi_post_inv= Psi_post_inv +np.eye(K)*1e-8
            Psi_post= np.linalg.inv(Psi_post_inv)
            #posterior mean of phi
            Phi_post= Psi_post @ (Psi_0_inv @ Phi_0 + XY)
            #posterior scale of sigma (s_post)
            E= Y -X@ Phi_post
            #add sum of squared residuals and prior deviation
            S_post= S_0 +(E.T @ E) +((Phi_post - Phi_0).T @ Psi_0_inv@(Phi_post-Phi_0))
            #ensure positive definite
            S_post= np.atleast_2d(S_post) +np.eye(N) *1e-10

            #posterior degrees of freedom
            nu_post= nu_0 +T
            #draw sigma from inverse wishart distribution
            Sigma_draws= stats.invwishart.rvs(df=nu_post, scale=S_post, size=n_draws)
            Sigma_draws= np.asarray(Sigma_draws)
            #handle single draw case dimensions
            if Sigma_draws.ndim == 2:
                Sigma_draws= Sigma_draws[None, :, :]
            self.sigma_draws= Sigma_draws

            #sample phi draws conditional on sigma 
            self.phi_draws= np.zeros((n_draws, K, N))
            #cholesky of feature-side scale matrix
            L_Psi= np.linalg.cholesky(Psi_post)
            for d in range(n_draws):
                #extract current sigma draw
                Sigma= np.atleast_2d(self.sigma_draws[d])
                #cholesky of error-side scale matrix
                L_Sigma= np.linalg.cholesky(Sigma)
                #draw standard normal matrix
                Z= np.random.standard_normal((K, N))
                #transform to matricvariate normal draw
                self.phi_draws[d]= Phi_post +L_Psi@ Z@ L_Sigma.T

            return self

    def shapley_params(self, data, target_idx):
        """
        extracts inputs for Shapley value calculation, including target lags"""
        #extract raw data for both targets and features
        target_cols= [c for c in data.columns if 'target_' in c]
        feature_cols= [c for c in data.columns if 'target_' not in c]        
        #get values as arrays
        Y_raw= data[target_cols].values
        X_raw= data[feature_cols].values
        n_obs= data.shape[0]        
        #recreate combined lag vector (match create_lags order)
        lags= []
        for lag in self.lag_indices:
            #find index relative to end of sample
            idx= n_obs -lag-1
            #order: target lags then exogenous lags
            lags.append(Y_raw[idx, :])
            lags.append(X_raw[idx, :])            
        #merge all lagged values
        X_combined= np.concatenate(lags)        
        #apply deduplication mask
        X_unique= X_combined[self.kept_indices]
        
        #generate names for all predictors
        all_names= []
        for lag in self.lag_indices:
            #label target lags
            for name in target_cols:
                all_names.append(f"{name}_lag{lag}")
            #label feature lags
            for name in feature_cols:
                all_names.append(f"{name}_lag{lag}")        
        #keep only names for unique columns
        final_names= [all_names[i] for i in self.kept_indices]
        
        #extract coefficients calc mean across posterior draws
        beta_mean_all= np.mean(self.phi_draws, axis=0) 
        #isolate current target equation
        beta_target= beta_mean_all[:, target_idx]        
        #split intercept and slopes
        intercept= beta_target[0]
        coeffs= beta_target[1:] 
        
        #package results
        x_series= pd.Series(X_unique, index=final_names)
        coeffs_dict= dict(zip(final_names, coeffs))
        
        return x_series, coeffs_dict, intercept

    def forecast(self, data):
        #extract predictor columns using target exclusion logic
        feature_cols= [c for c in data.columns if 'target_' not in c]
        target_cols= [c for c in data.columns if 'target_' in c]
        Y_raw= data[target_cols].values
        X_raw= data[feature_cols].values        
        #store lagged values for t-h logic
        lags= []
        n_obs= X_raw.shape[0]         
        #iterate lag structure
        for lag in self.lag_indices:
            #use data at time t to predict t+h
            idx_y= n_obs -lag-1  
            idx= n_obs - lag- 1                
            #validate sufficient history exists
            if idx_y < 0 or idx < 0:
                raise ValueError("not enough data in x_test to create lags for forecast.")
                
            #append target and exogenous blocks
            lags.append(Y_raw[idx_y, :]) 
            lags.append(X_raw[idx, :])
        #stack into horizontal vector
        X_combined= np.concatenate(lags)
        #remove duplicate lags
        X_unique= X_combined[self.kept_indices]
        #add constant term
        x_t= np.concatenate([[1.0], X_unique])        
        #get draws count for predictive simulation
        n_draws= self.phi_draws.shape[0]  
        preds= np.zeros((n_draws, self.n_vars))  
        
        #simulate future outcomes
        for i in range(n_draws):
            #get current cov matrix draw
            Sigma= np.atleast_2d(self.sigma_draws[i])  
            #draw multivariate shock
            noise= np.random.multivariate_normal(np.zeros(self.n_vars), Sigma) 
            #predict mean and add stochastic error
            preds[i, :]= x_t@self.phi_draws[i]+ noise
            
        return preds