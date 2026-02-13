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
    prior_params=None,
    fixed_lambda=0.02,
    start_eval=60,
    step=1,
    n_draws=600,
    burn_in=150,
    y_is_preshifted=False,
):
    """
    Rolling-origin CRPS evaluation for a FIXED hyperparameter set.

    y_is_preshifted=False means: data[target_col] is y_t, and realized is y_{t+h} => use t+h
    y_is_preshifted=True  means: data[target_col] at row t already equals y_{t+h} => use t
    """

    # auto-detect target column if not provided
    if target_col is None:
        target_cols = [c for c in data.columns if "target_" in c]
        if len(target_cols) != 1:
            raise ValueError(
                f"target_col not provided and found {len(target_cols)} target_ columns: {target_cols}"
            )
        target_col = target_cols[0]

    if target_col not in data.columns:
        target_cols = [c for c in data.columns if "target_" in c]
        raise KeyError(f"'{target_col}' not found. Available target columns: {target_cols}")

    T = len(data)
    if start_eval >= T - (0 if y_is_preshifted else horizon):
        raise ValueError(
            f"start_eval={start_eval} too large for T={T}, horizon={horizon}, y_is_preshifted={y_is_preshifted}. "
            f"Need start_eval < {T - (0 if y_is_preshifted else horizon)}."
        )

    scores = []

    last_origin = T if y_is_preshifted else (T - horizon)
    for t in range(start_eval, last_origin, step):
        train = data.iloc[:t].copy()

        # realized
        y_true = data.iloc[t][target_col] if y_is_preshifted else data.iloc[t + horizon][target_col]

        # Fit with fixed lambda (NO tuning inside)
        b = BVAR(lags=2, prior_type="natural_niw", prior_params=prior_params)
        b.fit(train, horizon=horizon, n_draws=n_draws, burn_in=burn_in, fixed_lambda=fixed_lambda)

        draws = b.forecast(train)[:, target_idx]
        scores.append(crps_from_samples(y_true, draws))

    return float(np.nanmean(scores))


def tune_natural_niw_by_crps(
    data,
    horizon=12,
    target_col=None,
    target_idx=0,
    start_eval=60,
    step=1,
    y_is_preshifted=False,
    n_draws=600,
    burn_in=150,
    # grids:
    lambda_grid=(0.005, 0.01, 0.02, 0.05, 0.08, 0.12),
    theta_grid=(0.001, 0.003, 0.01, 0.03, 0.1),
    alpha_decay_grid=(1.0, 2.0, 3.0),
    a3=100.0,
):
    """
    Returns (best_score, best_params_dict).
    best_params_dict contains: lambda, theta, alpha_decay, alpha(a3).
    """

    best_score = np.inf
    best_params = None

    for lam in lambda_grid:
        for theta in theta_grid:
            for a_decay in alpha_decay_grid:
                prior_params = {"theta": float(theta), "alpha_decay": float(a_decay), "alpha": float(a3)}

                score = rolling_crps_score(
                    data=data,
                    target_col=target_col,
                    target_idx=target_idx,
                    horizon=horizon,
                    prior_params=prior_params,
                    fixed_lambda=float(lam),
                    start_eval=start_eval,
                    step=step,
                    n_draws=n_draws,
                    burn_in=burn_in,
                    y_is_preshifted=y_is_preshifted,
                )

                if score < best_score:
                    best_score = score
                    best_params = {"lambda": float(lam), "theta": float(theta), "alpha_decay": float(a_decay), "alpha": float(a3)}

    return best_score, best_params

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
        self.lag_indices =[0, 1, 2]
        max_lag= self.h+2  #biggest lag
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

    def minnesota_dummies(self, a1, a2,a3, sigmas_y, sigmas_x): 
        """creates dummy observations: minnesota beliefs into data rows"""
        N= self.n_vars     #nr target variables
        K= self.n_features        #total features 
        L= len(self.lag_indices)   #nr lags
        #calc number of predictors
        n_preds= (K- 1) //L         
        #prior for Lags 
        n_dum_lags= K-1   #number of lagged dummy features
        yd_1= np.zeros((n_dum_lags, N))  #prior mean is zero for all lagged features
        xd_1= np.zeros((n_dum_lags, K))  #dummy features for lag priors
        #loop through lagged features to set dummy values based on minnesota logic
        for j_feat in range(n_dum_lags):
            #identify which lag and which predictor variable this feature corresponds to
            lag_pos=j_feat //n_preds  #lag position (0 for lag 0, 1 for lag 1, etc.)
            var_j_idx= j_feat %n_preds     #var index
            #get scaling factor for this lag based on minnesota formula
            k =self.lag_indices[lag_pos]
            k_scale= max(k, 1) #treat lag 0 as 1 for scaling to avoid div by zero
        
            col_idx =j_feat +1 #col index in dummy X (offset by 1 for intercept)
            
            #create dummy for each equation 
            if var_j_idx < N:
                # own lags i==j
                xd_1[j_feat, col_idx]= (k_scale*sigmas_x[var_j_idx]) /np.sqrt(a1)
            else:
                #other lags:use the avg sigma or the corresponding sigma for scaling
                s_jj = sigmas_x[var_j_idx] if var_j_idx < len(sigmas_x) else np.mean(sigmas_x)
                xd_1[j_feat, col_idx] = (k_scale * s_jj) / np.sqrt(a2)
        #prior for intercepts
        xd_2 =np.zeros((1, K))  #intercept dummy features
        xd_2[0, 0]= 1.0/np.sqrt(a3)  #prior tightness for intercepts
        yd_2= np.zeros((1, N))     #prior mean for intercepts (zero)        
        #sigma prior
        yd_sig=np.diag(sigmas_y)
        xd_sig= np.zeros((N, K))
        #combine all dummies by stacking vertically
        X_dum =np.vstack([xd_1, xd_2, xd_sig])
        Y_dum=np.vstack([yd_1, yd_2, yd_sig])
        
        return X_dum, Y_dum

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

    
    def log_ml_minnesota(self, params, X, Y, a3, sigmas_y, sigmas_x):
        """ Log Marginal Likelihood for Minnesota Prior """
        a1 = np.exp(params[0])
        a2 = np.exp(params[1]) # Relative cross-tightness
        
        X_dum, Y_dum = self.minnesota_dummies(a1, a2, a3, sigmas_y, sigmas_x)
        X_star = np.vstack([X_dum, X])
        Y_star = np.vstack([Y_dum, Y])
        
        # Standard Normal-Inverse Wishart Marginal Likelihood formula
        T_star, K = X_star.shape
        T = Y.shape[0]
        N = Y.shape[1]
        
        XX_star = X_star.T @ X_star + np.eye(K) * 1e-8
        L_xx = np.linalg.cholesky(XX_star)
        log_det_xx = 2 * np.sum(np.log(np.diag(L_xx)))
        
        Phi_post = np.linalg.solve(XX_star, X_star.T @ Y_star)
        S_post = (Y_star - X_star @ Phi_post).T @ (Y_star - X_star @ Phi_post)
        
        sign, log_det_s = np.linalg.slogdet(S_post)
        
        # Marginal Likelihood for the Natural Conjugate form
        # Log ML proportional to:
        log_ml = - (T * N / 2) * np.log(np.pi) + (N / 2) * (-log_det_xx) - ((T_star - K) / 2) * log_det_s
        return -log_ml # Return negative for minimization
 




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
        alpha= self.params.get('alpha_decay', 3.0)    
        #prior mean: identity for first own-lag, zeros otherwise
        Phi_0= np.zeros((K, N))         
        #prior tightness matrix
        psi_diag =np.zeros(K)        
        #intercept tightness
        psi_diag[0]= a3  
        #loop through lag indices to apply shrinkage
        for j, f_type in enumerate(self.feature_type_map):
            #column index in the full Phi matrix (including intercept->+1)
            col_idx= j+1             
          
            #use the specific sigma for this feature
            s_jj= sigmas_x[j]    
            #lag distance
            lag_dist=self.h        
            # Find if this column is target (0) or exogenous (1)
            f_type = self.feature_type_map[j]            
            if f_type== 0:
                #target lags (Standard Minnesota)
                psi_diag[col_idx]=(lam**2)/(s_jj**2* (lag_dist**alpha))
            else:
                #exogenous features (Extra Tightness to handle multicollinearity)
                psi_diag[col_idx]= (theta*lam**2)/ (s_jj**2 *(lag_dist**alpha))
        return Phi_0, np.diag(psi_diag)


    def fit(self, data, horizon=12, n_draws=2000, burn_in=500):
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
        #as create lags only for X, need seperate sigma calculation for the raw features
        sigmas_x_raw=[]
        #loop through each feature column in the raw X data to calculate its sigma
        for idx in range(X_raw_data.shape[1]):
            res=np.diff(X_raw_data[:, idx])  #use diff to get changes
            sigmas_x_raw.append(np.std(res) if len(res) > 0 else 0.1)   #avoid zero std if not enough data
        #to array
        sigmas_x_raw = np.array(sigmas_x_raw)
        #lag order to align with create lag function
        sigmas_one_block= np.concatenate([sigmas, sigmas_x_raw])
        L= len(self.lag_indices)  #number of lag blocks
        sigmas_tiled=np.tile(sigmas_one_block, L)        #tile the raw sigmas according to the lag structure
        
        #apply the same deduplication mask used in create_lags
        sigmas_all = sigmas_tiled[self.kept_indices]
        #Minnesota config
        #-----------------
        if 'minnesota' in self.prior_type:
            #def default params
            a3= self.params.get('alpha', 100.0)
            #initial guesses for log(a1) and log(a2)
            init_params=[np.log(0.01), np.log(0.15)] 
            #residuals by optimizing the log marginal likelihood with respect to a1 and a2 
            res =optimize.minimize(self.log_ml_minnesota, init_params, args=(X, Y, a3, sigmas, sigmas_all),bounds=[(np.log(0.01), np.log(0.3)), (np.log(0.01), np.log(0.5))])
            #get optimal a1 and a2 by exponentiating the log values
            #a1, a2=np.exp(res.x)
            a1=0.05
            a2=0.025
            #create dummy observations based on final lambda
            X_dum, Y_dum= self.minnesota_dummies(a1, a2,a3, sigmas, sigmas_all)
            
            
            #augment data: instead of making giant cov matrix-> add rows to data
            Y_star=np.vstack([Y_dum, Y])
            X_star=np.vstack([X_dum, X])
            #add jitter to diagonal to avoid multicolinearity proble
            XX= X_star.T @ X_star
            jitter= np.eye(K)*1e-4   #tiny nudge to make positive definite
            XX_stable =XX+jitter            
            #get precision matrix via cholesky for efficiency
            L_precision= np.linalg.cholesky(XX_stable)
            #posterior mean
            Phi_post=np.linalg.solve(XX_stable, X_star.T @ Y_star)
            #initialize storage for posterior draws
            phi_list= []
            #fix sigma by using estimate from the data
            Sigma =np.diag(sigmas**2)

            for i in range(n_draws +burn_in):
                #use fixed sigma for cholesky decomposition
                L_S=np.linalg.cholesky(Sigma)                
                #draw Phi
                Z = np.random.standard_normal((K, N))
                innovation= np.linalg.solve(L_precision.T, Z)@L_S.T
                Phi= Phi_post+innovation
                
                if i >= burn_in:
                    phi_list.append(Phi)
                    #store same Sigma every time for minnesota logic
                    self.sigma_draws.append(Sigma) 
            #store draws as arrays
            self.phi_draws=np.array(phi_list)
            self.sigma_draws=np.array(self.sigma_draws)

                
        #independent normal-inverse wishart prior
        #----------------------------------
        elif 'independent_niw' in self.prior_type:
            #hyperparameters
            a1=self.params.get('lambda', 0.2)   # own lag tightness
            a2= self.params.get('theta', 0.5)    #cross-variable tightness
            a3= self.params.get('alpha', 100.0) #intercept tightness
            n_iter =self.params.get('sampling', {}).get('n_draws', 2000)    #number of posterior draws
            burn_in= self.params.get('sampling', {}).get('burn_in', 500)  #burn-in period-> discard initial samples for convergence
                        
            #prior moments
            alpha_0=np.zeros(N*K)  #all zeros for WN
            V_alpha_0= np.zeros(N*K) #to store prior precision diag
            #loop through each equation and feature to set prior variances based on minnesota logic
            for n in range(N): 
                for k in range(K):
                    #get index in vectorized alpha
                    idx= n*K +k
                    #if feature is intercept, set variance based on a3, otherwise use minnesota logic for own vs cross lags
                    if k ==0: # Intercept
                        V_alpha_0[idx]= a3
                    else:
                        #logic to identify own lag vs cross lag: feature_idx is k-1 (ignoring intercept) var_j is the variable index (0 to N-1) this feature belongs to
                        var_j =(k-1)%N 
                        #if own lag                        
                        if var_j ==n:
                            #use a1 for own lags
                            V_alpha_0[idx]= a1**2
                        else:
                            #cross lags: use a2 and scale by sigma of variable j
                            V_alpha_0[idx]=(a1* a2)**2
            #define prior precision as inverse of variance
            V_alpha_0_inv =np.diag(1.0/V_alpha_0)
            #prior scale matrix
            S_0=np.diag(sigmas**2)
            #prior degrees of freedom
            nu_0= N+2
            #initialize for Gibbs sampling
            Sigma_current= np.diag(sigmas**2)   #start value for sigma
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
                V_alpha_post=np.linalg.inv(V_alpha_post_inv)   
                #calc weighted avf of prior and data
                data_term= (X.T@Y@Sigma_inv).flatten(order='F')
                alpha_hat= V_alpha_post @ (V_alpha_0_inv @alpha_0+data_term)
                #draw alpha from normal
                alpha_draw= np.random.multivariate_normal(alpha_hat, V_alpha_post)
                #convert vectorized alpha to (KxN)
                Phi_current=alpha_draw.reshape((K, N), order='F')

                #compute residuals
                residuals= Y -X @Phi_current
                #calc posterior scale matrix
                S_post= S_0+ residuals.T @residuals
                nu_post= nu_0+ T   #posterior degrees of freedom
                    
                #draw new Sigma from inverse-Wishart
                Sigma_current= stats.invwishart.rvs(df=nu_post, scale=S_post)

                #store draws after burn-in
                if it >= burn_in:
                    self.phi_draws[it-burn_in]= Phi_current
                    self.sigma_draws[it-burn_in]= Sigma_current
            

        # Natural Conjugate Normal-Wishart Prior
        # -------------------------------------
        elif 'natural_niw' in self.prior_type:
            #def default params  
            a3= self.params.get('alpha', 100.0)
            #want to find best lambda=a1=a2 by maximing log marginal likelihood
            #objective= lambda log_lam: -self.log_ml_natural(log_lam, X, Y, a3, sigmas, sigmas_all)  #objective to minimize (negative log marginal likelihood)
            #res=optimize.minimize_scalar(objective, bounds=(-2.0, 0.0), method='bounded')      #search for best lambda in log-space
            #best_lambda= np.exp(res.x)  #optimal lambda
            # Use Analytical Posterior instead of Dummies (much more stable for large feature sets)
            #Phi_0, Psi_0 = self.natural_moments(N, best_lambda, a3, sigmas, sigmas_all)
            objective = lambda log_lam: -self.log_ml_natural(log_lam, X, Y, a3, sigmas, sigmas_all)
            res = optimize.minimize_scalar(objective, bounds=(-6.0, 0.0), method='bounded')  # widen
            lam_used = np.exp(res.x)
            best = (np.inf, None)
            target_cols = [c for c in data.columns if c.startswith("target_") or "target_" in c]

            for lam in [0.005, 0.01, 0.02, 0.05, 0.08, 0.12]:
                for theta in [0.001, 0.003, 0.01, 0.03, 0.1]:
                    for a_decay in [1.0, 2.0, 3.0]:
                        params = {"theta": theta, "alpha_decay": a_decay, "alpha": 100.0}
                        score = rolling_crps_score(
                            data=data,
                            target_col=target_cols[0]   # if you only have one target series
                            target_idx=0,                    # equation index
                            horizon=12,
                            prior_params=params,
                            fixed_lambda=lam,
                            start_eval = max(horizon + 24, int(0.6 * len(data)))
,                  # e.g. after 10 years monthly
                            step=1,
                            n_draws=800,
                            burn_in=200)
                        if score < best[0]:
                            best = (score, (lam, theta, a_decay))




            Phi_0, Psi_0 = self.natural_moments(N, lam_used, a3, sigmas, sigmas_all)

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