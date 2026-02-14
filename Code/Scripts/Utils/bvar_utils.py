import numpy as np
from scipy import stats, optimize, special
import pandas as pd


import numpy as np

def crps_from_samples(y, samples):
    """
    Proper CRPS for an empirical predictive distribution (samples).
    """
    s = np.asarray(samples).ravel()
    if s.size == 0 or np.isnan(y):
        return np.nan
    term1 = np.mean(np.abs(s - y))
    term2 = 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))
    return float(term1 - term2)


def rolling_crps_score(
    data,
    target_col=None,
    target_idx=0,
    horizon=12,
    prior_type="natural_niw",
    prior_params=None,
    fixed_lambda=0.02,
    start_eval=60,
    step=1,
    n_draws=600,
    burn_in=150,
    y_is_preshifted=False,
):
    # auto-detect target column if not provided
    if target_col is None:
        target_cols = [c for c in data.columns if "target_" in c]
        if len(target_cols) != 1:
            raise ValueError(f"target_col not provided and found {len(target_cols)} target_ columns.")
        target_col = target_cols[0]

    T = len(data)
    scores = []
    last_origin = T if y_is_preshifted else (T - horizon)

    for t in range(start_eval, last_origin, step):
        train = data.iloc[:t].copy()
        y_true = data.iloc[t][target_col] if y_is_preshifted else data.iloc[t + horizon][target_col]

        # Fit with fixed lambda (NO tuning inside the loop)
        b = BVAR(lags=2, prior_type=prior_type, prior_params=prior_params)
        b.fit(train, horizon=horizon, n_draws=n_draws, burn_in=burn_in, fixed_lambda=fixed_lambda)

        draws = b.forecast(train)[:, target_idx]
        scores.append(crps_from_samples(y_true, draws))

    return float(np.nanmean(scores))



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
        # Initialize Cache for tuning if it doesn't exist
        self._mn_tuned_cache = {}


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
        
        #define want lags 0, 1, 2
        self.lag_indices =[0, 1]
        max_lag= self.h+1  #biggest lag
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
            # k is the index in the original huge stack
            # mod block_size gives the variable index
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

    def get_minnesota_precision(self, lam, theta, alpha, a3, sigmas_x):
        """
        Constructs the Prior Precision Matrix (Diagonal) analytically.
        Precision = Inverse of Variance.
        """
        K = self.n_features
        # Initialize Prior Precision Diagonal (High precision = Tight prior near 0)
        # Variance ~ lambda^2. Precision ~ 1/lambda^2.
        
        precision_diag = np.zeros(K)
        
        # Intercept Precision (Index 0)
        # Variance = a3. Precision = 1/a3.
        precision_diag[0] = 1.0 / a3
        
        block_size = self.n_vars + self.n_exog
        
        for j, f_type in enumerate(self.feature_type_map):
            col_idx = j + 1
            orig_j = int(self.kept_indices[j])
            
            # Use sigma of the predictor
            s_jj = sigmas_x[j]
            
            # Lag distance
            lag_pos = orig_j // block_size
            lag = self.lag_indices[lag_pos]
            
            # Decay factor (k^alpha)
            # Higher alpha -> Distant lags have less variance -> Higher Precision
            lag_decay = max(1, lag) ** alpha
            if f_type == 0: # Target lags
                lag_decay_target = (self.h + max(1, lag)) ** alpha
                # Variance = (lambda / lag^alpha * sigma)^2 ... simplified
                # Let's use standard: Var = (lambda^2) / (s_jj^2 * k^2alpha)
                # Precision = (s_jj^2 * k^2alpha) / lambda^2
                precision_diag[col_idx] = (s_jj**2 * lag_decay_target) / (lam**2)
            else:
                # Exogenous: Tighter by theta
                # Variance = (lambda * theta)^2 / ...
                # Precision = ... / (lambda*theta)^2
                precision_diag[col_idx] = (s_jj**2 * lag_decay) / ((lam * theta)**2)
                
        return np.diag(precision_diag)


    def independent_priors(self, N, K, a1, a2, a3, sigmas):
        """
        Constructs the vectorized priors alpha_0 (NK x 1) and V_alpha_0 (NK x NK)
        """
        # 1. Prior Mean alpha_0 (vec(Phi_0))
        # Usually 0, or 1 for the first lag of own variable (Random Walk)
        Phi_0 = np.zeros((K, N))
        # Example: Assume White Noise (prior mean 0). 
        # For Random Walk: for i in range(N): Phi_0[1+i, i] = 1.0
        alpha_0 = Phi_0.flatten(order='F')

        # 2. Prior Covariance V_alpha_0 (NK x NK)
        # We construct this as a diagonal matrix to allow independent shrinkage
        V_diag = np.zeros(K * N)
        
        for n in range(N): # For each equation
            for k in range(K): # For each feature
                idx = n * K + k
                if k == 0: # Intercept
                    V_diag[idx] = (sigmas[n]**2) *a3
                else:
                    # Determine which variable 'j' and which lag 'p' this feature belongs to
                    # This logic depends on how create_lags organized X.
                    # Simplified Minnesota-style logic:
                    V_diag[idx] = (a1**2) # Simplified for custom lag structure
        
        V_alpha_0 = np.diag(V_diag)
        V_alpha_0_inv = np.diag(1.0 / V_diag)
        
        return alpha_0, V_alpha_0_inv

    def _compute_posterior_minnesota(self, X, Y, best_lam, best_theta, best_decay, a3, sigmas, sigmas_all):
        """
        HELPER FUNCTION: Handles both Analytical and Dummies logic.
        Returns: Phi_post, S_post, Cholesky_Precision (for sampling)
        """
        N = self.n_vars
        K = self.n_features
        T = X.shape[0]
        
        # Define Prior S_0 (Scale Matrix) - Common to both
        S_0 = np.diag(sigmas**2)

        if self.implementation_type == 'analytical':
            # ---------------- ANALYTICAL ----------------
            # 1. Get Moments
            self.params['theta'] = best_theta
            self.params['alpha_decay'] = best_decay
            Phi_0, Psi_0 = self.natural_moments(N, best_lam, a3, sigmas, sigmas_all)
            
            # 2. Posterior Precision
            Psi_0_inv = np.linalg.inv(Psi_0) # This is Prior Variance
            # Note: Your natural_moments returns Psi_0 as VARIANCE or PRECISION? 
            # Looking at natural_moments: it calculates (lam**2)/sigma... that is Variance.
            # So Psi_0 in natural_moments is actually V_prior.
            # Precision = inv(V_prior).
            
            # Let's verify natural_moments logic: 
            # psi_diag[col] = (lam**2)... This is Variance.
            # So Psi_0 is Prior Variance Matrix (V_0).
            V_0 = Psi_0 
            V_0_inv = np.diag(1.0 / np.diag(V_0)) # Prior Precision
            
            XX = X.T @ X
            V_post_inv = V_0_inv + XX # Posterior Precision
            
            # Stability Jitter
            V_post_inv += np.eye(K) * 1e-6
            
            # 3. Posterior Mean
            # Phi_post = V_post @ (V_0_inv @ Phi_0 + X'Y)
            # Phi_0 is zero, so:
            Phi_post = np.linalg.solve(V_post_inv, X.T @ Y)
            
            # 4. Posterior Scale S_post
            # S_post = S_0 + Y'Y + Phi_0' V_0_inv Phi_0 - Phi_post' V_post_inv Phi_post
            term_data = Y.T @ Y
            term_post = Phi_post.T @ V_post_inv @ Phi_post
            S_post = S_0 + term_data - term_post
            
            return Phi_post, S_post, V_post_inv

        else:
            # ---------------- DUMMIES ----------------
            # 1. Get Dummies
            X_dum, Y_dum = self.minnesota_dummies(best_lam, best_theta, a3, best_decay, sigmas, sigmas_all)
            
            # 2. Augment Data
            X_star = np.vstack([X_dum, X])
            Y_star = np.vstack([Y_dum, Y])
            
            # 3. Posterior via OLS on augmented data
            XX_star = X_star.T @ X_star + np.eye(K) * 1e-6
            XY_star = X_star.T @ Y_star
            
            Phi_post = np.linalg.solve(XX_star, XY_star)
            
            # 4. Posterior Scale
            # S_post = S_0 + SSE_star
            # Note: We must ensure S_0 isn't double counted.
            # In purely dummy approaches, S_0 is often encoded in dummies. 
            # Here we add it explicitly to match NIW.
            E_star = Y_star - X_star @ Phi_post
            S_post = S_0 + E_star.T @ E_star
            
            return Phi_post, S_post, XX_star




    def log_ml_natural(self, log_lambda, X, Y, a3, sigmas, sigmas_x):
        """compute log marginal likelihood for natural niw prior"""
        #define parameters
        lam= np.exp(log_lambda)
        T, N=Y.shape  #get dimensions
        K = self.n_features
        #get prior based on lambda
        Phi_0, Psi_0= self.natural_moments(N, lam, a3, sigmas, sigmas_x)
        Psi_0_inv= np.diag(1.0/np.diag(Psi_0))  #work with precision (inverse) for stability
        #posterior parameters
        XX= X.T @X
        Psi_post_inv= Psi_0_inv +XX  #posterior precision
        #log determinants for psi
        L_post_inv= np.linalg.cholesky(Psi_post_inv)     
        log_det_Psi_post_inv= -2.0*np.sum(np.log(np.diag(L_post_inv)))
        log_det_Psi_0= np.sum(np.log(np.diag(Psi_0)))
        log_det_Psi_post = -log_det_Psi_post_inv #since Psi_post is inverse of Psi_post_inv
        #posterior mean
        rhs= Psi_0_inv @Phi_0+X.T@ Y  #right-hand side of posterior mean formula
        Phi_post= np.linalg.solve(Psi_post_inv, rhs)    
        #prior scale matrix
        S_0= np.diag(sigmas**2)
        #get posterior scale matrix via helpers
        term_data= Y.T @Y  
        term_prior= Phi_0.T @ Psi_0_inv @Phi_0
        term_post= Phi_post.T @ Psi_post_inv @Phi_post
        S_post= S_0 +term_data +term_prior -term_post
        #get prior degrees of freedom
        nu_0= N+2
        nu_post= nu_0+ T    #posterior degrees of freedom
        #log determinant of S_0
        log_det_S_0= np.sum(np.log(np.diag(S_0)))
        #log determinant of S_post
        sign, log_det_S_post= np.linalg.slogdet(S_post)
        #compute log marginal likelihood
        term_cov=(N/2.0)*(log_det_Psi_post - log_det_Psi_0)  #covariance terms from prior and posterior
        term_scale= (nu_0/2.0)*log_det_S_0-(nu_post/2.0)*log_det_S_post  #scale terms from prior and posterior  
        const_gamma=special.multigammaln(nu_post/2.0, N)- special.multigammaln(nu_0/2.0, N)  #gamma function terms
        log_ml= term_cov +term_scale + const_gamma- (T*N/2.0)*np.log(np.pi)
        return log_ml
    
    def natural_moments(self, N, lam, a3, sigmas_y, sigmas_x):
        """ constructs prior moments for natural conjugate normal-inverse wishart prior"""
        #get dimensions
        K= self.n_features
        #manually control how much we trust exogenous vs target lags
        theta = self.params.get('theta', 0.01) 
        # manually control lag decay (higher= distant lags shrink faster)
        alpha= self.params.get('alpha_decay', 2.0)    
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

    def _estimate_sigma_for_forecast(self, X, Y, Phi_post_all, sigmas):
        """Returns Sigma used for forecast noise in Minnesota mode.
        - diag: diagonal from univariate sigmas (your current behavior)
        - ols_shrink: OLS residual covariance shrunk toward diag
        """
        mode = self.params.get("sigma_mode", "diag")
        if mode == "diag":
            return np.diag(sigmas**2)

        if mode == "ols_shrink":
            E = Y - X @ Phi_post_all
            Sigma_ols = (E.T @ E) / max(1, E.shape[0] - 1)
            Sigma_diag = np.diag(np.diag(Sigma_ols))
            rho = float(self.params.get("sigma_shrink", 0.2))  # 0 -> diag only, 1 -> full OLS
            Sigma = (1 - rho) * Sigma_diag + rho * Sigma_ols
            # jitter for PSD
            Sigma += np.eye(Sigma.shape[0]) * 1e-8
            return Sigma

        raise ValueError(f"Unknown sigma_mode: {mode}")


    def fit(self, data, horizon=12, n_draws=2000, burn_in=500, fixed_lambda=None):
        """estimate bvar
        """
        #create lagged matrices calling fct
        X, Y= self.create_lags(data, horizon=horizon)
        #get dimensions
        T, N=Y.shape    #nr of equations and obs.
        K=self.n_features  #nr of features

        #calc res variances from univariate AR for targets
        sigmas=[]  #to store scales
        for idx in range(N):
            #fit univariate AR to get residual std
            res= np.diff(Y[:, idx])
            sigmas.append(np.std(res) if len(res)>0 else 0.1)
        sigmas= np.array(sigmas)
 
        #extract feature columns before they were lagged
        feature_cols= [c for c in data.columns if 'target_' not in c]
        X_raw_data= data[feature_cols].values  #get raw feature data (without lags) 
        feature_cols_raw = X[:, 1:] # remove intercept
        sigmas_x_all = []
        for idx in range(feature_cols_raw.shape[1]):
            res = np.diff(feature_cols_raw[:, idx])
            sigmas_x_all.append(np.std(res) if len(res) > 0 else 0.1)
        sigmas_all = np.array(sigmas_x_all)
        #as create lags only for X, need seperate sigma calculation for the raw features
        #sigmas_x_raw=[]
        #loop through each feature column in the raw X data to calculate its sigma
        #for idx in range(X_raw_data.shape[1]):
         #   res=np.diff(X_raw_data[:, idx])  #use diff to get changes
          #  sigmas_x_raw.append(np.std(res) if len(res) > 0 else 0.1)   #avoid zero std if not enough data
        #to array
        #sigmas_x_raw = np.array(sigmas_x_raw)
        #lag order to align with create lag function
        #sigmas_one_block= np.concatenate([sigmas, sigmas_x_raw])
        #L= len(self.lag_indices)  #number of lag blocks
        #sigmas_tiled=np.tile(sigmas_one_block, L)        #tile the raw sigmas according to the lag structure
        
        #apply the same deduplication mask used in create_lags
        #sigmas_all = sigmas_tiled[self.kept_indices]
        #Minnesota config
        #-----------------
        if 'minnesota' in self.prior_type:

            # --- Minnesota hyperparams ---
            # (keep your existing key if you want; I’ll leave it as your current one)
            a3 = float(self.params.get('alpha', 100.0))  # intercept prior variance in your naming
            
            # --- 1) Tune ONCE + cache results ---
            # Using the instance-level cache dictionary created in __init__
            cache_key = ("minnesota", horizon, int(self.n_features), int(self.n_vars))

            if fixed_lambda is not None:
                best_lam = float(fixed_lambda)
                best_theta = float(self.params.get('theta', 0.01))
                best_decay = float(self.params.get('alpha_decay', 2.0))
            else:
                if cache_key in self._mn_tuned_cache:
                    best_lam, best_theta, best_decay = self._mn_tuned_cache[cache_key]
                else:
                    # --- LIGHT tuning (coarse) ---
                    lambda_grid=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]  #best lambda=a1=a2 (in accordance with natural niw characteristics)
                    decay_grid= [2.0, 4.0, 6.0]  #aggressive lag decay
                    theta_grid=[0.001, 0.003, 0.01, 0.03, 0.1] 

                    # Cheaper tuning settings
                    start_eval = max(horizon + 24, int(0.75 * len(data)))
                    step_tune = 4
                    n_draws_tune = 150
                    burn_in_tune = 50

                    target_cols = [c for c in data.columns if "target_" in c]
                    target_col = target_cols[0]

                    best_score = np.inf
                    best_params_tuple = (0.2, 0.5, 2.0)

                    for lam in lambda_grid:
                        for theta in theta_grid:
                            for dec in decay_grid:
                                p_params = {
                                    "theta": float(theta),
                                    "alpha_decay": float(dec),
                                    "alpha": float(a3),
                                }
                                # Call with fixed_lambda to avoid infinite recursion
                                score = rolling_crps_score(
                                    data=data,
                                    target_col=target_col,
                                    target_idx=0,
                                    horizon=horizon,
                                    prior_type="minnesota",
                                    prior_params=p_params,
                                    fixed_lambda=float(lam),
                                    start_eval=start_eval,
                                    step=step_tune,
                                    n_draws=n_draws_tune,
                                    burn_in=burn_in_tune,
                                )
                                if score < best_score:
                                    best_score = score
                                    best_params_tuple = (float(lam), float(theta), float(dec))

                    best_lam, best_theta, best_decay = best_params_tuple
                    
                    # Store in cache
                    self._mn_tuned_cache[cache_key] = (best_lam, best_theta, best_decay)
                    self.params.update({'lambda': best_lam, 'theta': best_theta, 'alpha_decay': best_decay})

            # --- 2) Estimation (Equation-by-Equation Analytical Ridge) ---
            Phi_post_all = np.zeros((K, N))
            V_post_list = []

            XX = X.T @ X
            
            block_size = self.n_vars + self.n_exog
            use_h = False # Standard minnesota usually doesn't shift by h, but set True if desired

            for i in range(N):
                # Construct Diagonal Penalty Matrix P for Equation i
                P_diag = np.zeros(K)
                P_diag[0] = 1.0 / a3 # Intercept Precision

                for feat_j, f_type in enumerate(self.feature_type_map):
                    col_idx = feat_j + 1 # Offset for intercept

                    orig_k = int(self.kept_indices[feat_j])
                    lag_num = self.lag_indices[orig_k // block_size]
                    var_idx = self.feature_var_indices[feat_j]

                    sigma_j = sigmas_all[feat_j] # Sigma of the predictor
                    sigma_i = sigmas[i]          # Sigma of the dependent variable

                    # Lag Decay Function
                    if f_type == 0:
                        lag_dist = (self.h + max(1, lag_num)) if use_h else max(1, lag_num)
                    else:
                        lag_dist = max(1, lag_num)

                    lag_func = (lag_dist ** best_decay)

                    # Ridge Penalty Calculation: P = sigma_i^2 * Precision
                    # 1. Own Lag: Var = (lam / k^alpha)^2 
                    #    P = sigma_i^2 / Var = sigma_i^2 * k^2alpha / lam^2
                    # 2. Cross Lag: Var = (lam * theta * sigma_i / sigma_j / k^alpha)^2
                    #    P = sigma_i^2 / Var = (sigma_j * k^alpha / (lam * theta))^2  (sigma_i cancels)
                    
                    if f_type == 0 and var_idx == i:
                        # Own Lag
                        ridge_penalty = (sigma_i**2) * ((lag_func / best_lam)**2)
                    else:
                        # Cross Lag or Exogenous
                        ridge_penalty = ((sigma_j * lag_func) / (best_lam * best_theta))**2

                    P_diag[col_idx] = ridge_penalty

                # Posterior Precision = X'X + P
                Post_Precision_i = XX + np.diag(P_diag) + np.eye(K) * 1e-6

                # Solve for beta: (X'X + P) beta = X'y
                L_i = np.linalg.cholesky(Post_Precision_i)
                w = np.linalg.solve(L_i, X.T @ Y[:, i])
                phi_i = np.linalg.solve(L_i.T, w)

                Phi_post_all[:, i] = phi_i

                # Covariance for sampling: (X'X + P)^-1
                L_inv = np.linalg.solve(L_i, np.eye(K))
                Unscaled_Cov = L_inv.T @ L_inv
                V_post_list.append(Unscaled_Cov)

            # --- 3) Sampling ---
            # Fixed Sigma for forecast noise
            Sigma_fixed = np.diag(sigmas**2)

            # Create sigma draws (identical for every draw)
            self.sigma_draws = np.repeat(Sigma_fixed[None, :, :], n_draws, axis=0)

            # Vectorized sampling of Phi draws
            self.phi_draws = np.empty((n_draws, K, N))

            for i in range(N):
                # Posterior covariance for equation i coefficients:
                # Cov_i = sigma_i^2 * V_post_list[i] (Bayesian Linear Regression Result)
                cov_i = (sigmas[i]**2) * V_post_list[i]

                # numerical jitter
                cov_i = cov_i + np.eye(K) * 1e-10

                # Draw coefficients
                # draws = mean + Z @ L.T
                L = np.linalg.cholesky(cov_i)
                Z = np.random.standard_normal((n_draws, K))
                self.phi_draws[:, :, i] = Phi_post_all[:, i][None, :] + Z @ L.T
                

                
        #independent normal-inverse wishart prior
        #----------------------------------
        elif 'independent_niw' in self.prior_type:
            #hyperparameters
            a1=self.params.get('lambda', 0.01)   # own lag tightness
            a2= self.params.get('theta', 0.01)    #cross-variable tightness
            a3= self.params.get('alpha', 100.0) #intercept tightness
            decay = self.params.get('alpha_decay', 2.0)
            n_iter =self.params.get('sampling', {}).get('n_draws', 2000)    #number of posterior draws
            burn_in= self.params.get('sampling', {}).get('burn_in', 500)  #burn-in period-> discard initial samples for convergence
                        
            #prior moments
            alpha_0=np.zeros(N*K)  #all zeros for WN
            V_alpha_0_diag= np.zeros(N*K) #to store prior precision diag
            block_size = self.n_vars + self.n_exog # original block size before deduplication
            #loop through each equation and feature to set prior variances based on minnesota logic
            Phi_0 = np.zeros((K, N))
            # Determine which feature is the 1st lag of the dependent variable
            # Usually: 1st block after intercept is Lag 0/1. 
            # We want the 'own-lag' to have a prior mean of 1.
            for n in range(N):
                # Search for the own-lag index in your feature mapping
                for k_search in range(1, K):
                    feat_idx = k_search - 1
                    if self.feature_type_map[feat_idx] == 0 and self.feature_var_indices[feat_idx] == n:
                        # Find the smallest lag (usually lag 1)
                        orig_k = int(self.kept_indices[feat_idx])
                        if self.lag_indices[orig_k // block_size] == 1: # Lag 1
                            Phi_0[k_search, n] = 1.0 # Set RW prior
                            break
            
            alpha_0 = Phi_0.flatten(order='F')

            # 2. Corrected Prior Variance Scaling
            for n in range(N): 
                for k in range(K):
                    idx = n * K + k
                    sigma_i = sigmas[n]
                    
                    if k == 0: 
                        # Intercept logic: NO f_type needed here
                        V_alpha_0_diag[idx] = (sigma_i**2) * a3
                    else:
                        # LAG/FACTOR logic: Define variables here
                        feat_j = k - 1
                        f_type = self.feature_type_map[feat_j]
                        orig_k = int(self.kept_indices[feat_j])
                        lag_num = self.lag_indices[orig_k // block_size]
                        var_idx = self.feature_var_indices[feat_j]
                        
                        sigma_j = sigmas_all[feat_j]
                        lag_func = (max(1, lag_num) ** decay)

                        # Now perform the check inside this scope
                        if f_type == 0 and var_idx == n:
                            # Own Lags
                            V_alpha_0_diag[idx] = (a1 / lag_func)**2
                        elif f_type == 0 and var_idx != n:
                            # Cross Lags
                            V_alpha_0_diag[idx] = ((a1 * a2 * sigma_i) / (sigma_j * lag_func))**2
                        else:
                            # PCA FACTORS (Exogenous block)
                            V_alpha_0_diag[idx] = ((a1 * a2 * sigma_i) / (sigma_j * lag_func))**2
            #define prior precision as inverse of variance
            V_alpha_0_inv =np.diag(1.0/V_alpha_0_diag)
            
            #prior scale matrix
            S_0=np.diag(sigmas**2)
            #prior degrees of freedom
            nu_0= N+2
            #initialize for Gibbs sampling
            Sigma_current = np.diag(sigmas**2)
            Sigma_current = np.atleast_2d(Sigma_current)  # ensure (N,N), also for N=1
  #start value for sigma
            #initialize storage for draws: vec bzw matrix of zeros
            self.phi_draws = np.zeros((n_iter - burn_in, K, N))
            self.sigma_draws = np.zeros((n_iter - burn_in, N, N))
            #X'X for precision update
            XX= X.T @X


            #iterate through draws to sample from posterior: gibbs sampling
            for it in range(n_iter):
                #take inverse of current sigma
                Sigma_inv= np.linalg.inv(Sigma_current)   
                #capture independent flexibility (posterior cov)
                V_alpha_post_inv=V_alpha_0_inv +np.kron(Sigma_inv,XX)
                # 2. Add a tiny bit of jitter to ensure it is positive definite
                V_alpha_post_inv += np.eye(V_alpha_post_inv.shape[0]) * 1e-9
                #use Cholesky for stability in sampling
                # 3. Use Cholesky on the PRECISION matrix
                L_upper = np.linalg.cholesky(V_alpha_post_inv).T 
                #calc weighted avf of prior and data
                data_term= (X.T@Y@Sigma_inv).flatten(order='F')
                rhs = V_alpha_0_inv @ alpha_0 + data_term
                # Solve L_upper.T @ L_upper @ alpha_hat = rhs
                alpha_hat = np.linalg.solve(L_upper, np.linalg.solve(L_upper.T, rhs))
                # 5. Draw alpha (Standard Normal Z)
                Z = np.random.standard_normal(N * K)
                # alpha_draw = alpha_hat + (L_upper^-1 @ Z)
                alpha_draw = alpha_hat + np.linalg.solve(L_upper, Z)

                # 6. Reshape
                Phi_current = alpha_draw.reshape((K, N), order='F')

                #compute residuals
                residuals= Y -X @Phi_current
                #calc posterior scale matrix
                S_post= S_0+ residuals.T @residuals
                nu_post= nu_0+ T   #posterior degrees of freedom
                    
                #draw new Sigma from inverse-Wishart
                Sigma_current = stats.invwishart.rvs(df=nu_post, scale=S_post)
                Sigma_current = np.atleast_2d(Sigma_current)  # SciPy returns scalar when N=1


                #store draws after burn-in
                if it >= burn_in:
                    self.phi_draws[it-burn_in]= Phi_current
                    self.sigma_draws[it-burn_in]= Sigma_current
            

        # Natural Conjugate Normal-Wishart Prior
        # -------------------------------------
        elif 'natural_niw' in self.prior_type:
            #def default params  
            a3= self.params.get('alpha', 100.0)
            #if not tune, set default
            if fixed_lambda is not None:
                best_lam= fixed_lambda
                best_theta= self.params.get('theta', 0.01)
                best_decay= self.params.get('alpha_decay', 2.0)
            else:
                #grids for crps tuning
                lambda_grid=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]  #best lambda=a1=a2 (in accordance with natural niw characteristics)
                alpha_decay_grid= [2.0, 4.0, 6.0]  #aggressive lag decay
                theta_grid=[0.001, 0.003, 0.01, 0.03, 0.1] 
                #at least 2 years to evaluate
                min_eval_window=24 
                #use last 40% for validation
                calc_start = int(len(data)*0.6)
                #ensure start date of evaluation is not too small
                start_eval= max(horizon+24, calc_start)
                if start_eval>= len(data)-horizon:
                    #fallback if not enough data
                    start_eval= len(data)- horizon-1
                #initialize best to infinity
                best_score = np.inf
                #set a fallback tuple
                best_params_tuple=(0.05, 1.0)
                #identify target cols 
                target_cols=[c for c in data.columns if c.startswith("target_") or "target_" in c]
                #grid search: loop through all 3 params
                for lam in lambda_grid:
                    for theta in theta_grid:
                        for a_decay in alpha_decay_grid:
                            #initialize prior dict
                            prior_params= {"theta": theta, "alpha_decay": a_decay, "alpha": a3}

                                # Use fewer draws/shorter chain for tuning speed
                            score = rolling_crps_score(data=data, target_col=target_cols[0], target_idx=0, horizon=horizon, prior_params=prior_params,
                                    fixed_lambda=lam, start_eval=max(horizon + 24, int(0.6 * len(data))), step=4, n_draws=400,  burn_in=100)
                            if score < best_score:
                                best_score = score
                                best_params_tuple = (lam, theta, a_decay)
                #get best params
                best_lam, best_theta, best_decay = best_params_tuple
                self.params['lambda'] = best_lam
                self.params['theta'] = best_theta
                self.params['alpha_decay'] = best_decay
            #generate final posterior
            Phi_0, Psi_0 = self.natural_moments(N, best_lam, a3, sigmas, sigmas_all)

            Psi_0_inv = np.linalg.inv(Psi_0)
            
            # Posterior Precision
            XX = X.T @ X
            Psi_post_inv = Psi_0_inv + XX + np.eye(K) * 1e-2 # Stability Jitter
            Psi_post = np.linalg.inv(Psi_post_inv)
            
            # Posterior Mean
            Phi_post = Psi_post @ (Psi_0_inv @ Phi_0 + X.T @ Y)
            
            # Posterior Scale Matrix (S_post)
            S_0 = np.diag(sigmas**2)
            # S_post = S_0 + (Y - X@Phi_post)'(Y - X@Phi_post) + (Phi_post-Phi_0)'Psi_0_inv(Phi_post-Phi_0)
            term_data = Y.T @ Y
            term_prior = Phi_0.T @ Psi_0_inv @ Phi_0
            term_post = Phi_post.T @ Psi_post_inv @ Phi_post
            S_post = S_0 + term_data + term_prior - term_post
            
            nu_post = N + 2 + T

            # Posterior Sampling
            self.sigma_draws = stats.invwishart.rvs(df=nu_post, scale=S_post, size=n_draws)
            self.phi_draws = np.zeros((n_draws, K, N))
            L_Psi = np.linalg.cholesky(Psi_post)
            
            for draw in range(n_draws):
                L_Sigma = np.linalg.cholesky(self.sigma_draws[draw])
                Z = np.random.standard_normal((K, N))
                self.phi_draws[draw] = Phi_post + L_Psi @ Z @ L_Sigma.T
            
        return self
    
    def shapley_params(self, data, target_idx):
        """
        Extracts inputs for Shapley value calculation, including target lags.
        """
        # 1. Extract raw data for both targets and features
        target_cols = [c for c in data.columns if 'target_' in c]
        feature_cols = [c for c in data.columns if 'target_' not in c]
        
        Y_raw = data[target_cols].values
        X_raw = data[feature_cols].values
        n_obs = data.shape[0]
        
        # 2. Recreate the combined lag vector (MUST match create_lags order)
        lags = []
        for lag in self.lag_indices:
            idx = n_obs - lag - 1
            # Order: Target Lags then Exogenous Lags
            lags.append(Y_raw[idx, :])
            lags.append(X_raw[idx, :])
            
        X_combined = np.concatenate(lags)
        
        # 3. Apply deduplication mask (This is where the error was happening)
        X_unique = X_combined[self.kept_indices]
        
        # 4. Generate names for all predictors (Targets + Features)
        all_names = []
        for lag in self.lag_indices:
            # Target names for this lag
            for name in target_cols:
                all_names.append(f"{name}_lag{lag}")
            # Feature names for this lag
            for name in feature_cols:
                all_names.append(f"{name}_lag{lag}")
        
        # Filter names to match unique columns
        final_names = [all_names[i] for i in self.kept_indices]
        
        # 5. Extract Coefficients
        # phi_draws shape: (draws, features, equations)
        beta_mean_all = np.mean(self.phi_draws, axis=0) 
        beta_target = beta_mean_all[:, target_idx]
        
        intercept = beta_target[0]
        coeffs = beta_target[1:] # Everything after intercept
        
        # 6. Package results
        x_series = pd.Series(X_unique, index=final_names)
        coeffs_dict = dict(zip(final_names, coeffs))
        
        return x_series, coeffs_dict, intercept
    
    def forecast(self, data):
        #extract only the predictor columns for create lags logic
        feature_cols= [c for c in data.columns if 'target_' not in c]
        target_cols = [c for c in data.columns if 'target_' in c]
        Y_raw = data[target_cols].values
        X_raw= data[feature_cols].values
        #initialize list to store lagged values
        lags= []
        n_obs= X_raw.shape[0] #number of observations in the input data
        #extract current values and lags
        for lag in self.lag_indices:
            #need to shift back the y values to not leak data for lags
            idx_y = n_obs - (lag + self.h) - 1
            idx= n_obs- lag-1   #index to extract the correct lagged value
            lags.append(Y_raw[idx_y, :])  #first target to match order of lag creation
            lags.append(X_raw[idx, :])
        #combine into 1 horizontal vector
        X_combined=np.concatenate(lags)
        #apply same deduplication used during training
        X_unique= X_combined[self.kept_indices]
        #add intercept
        x_t =np.concatenate([[1.0], X_unique])
        # -> x_t size should be 28, matching phi_draws
        n_draws= self.phi_draws.shape[0]  #number of posterior draws
        preds=np.zeros((n_draws, self.n_vars))  #to store predictions for each draw
        #loop through draws to generate predictive distribution
        for i in range(n_draws):
            noise=np.random.multivariate_normal(np.zeros(self.n_vars), self.sigma_draws[i])  #noise from current sigma draw
            #matrix multiplication to get mean prediction + add noise for uncertainty
            preds[i, :]= x_t@self.phi_draws[i]+ noise
            
        return preds