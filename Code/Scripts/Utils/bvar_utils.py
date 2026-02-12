import numpy as np
from scipy import stats, optimize, special
import pandas as pd

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
        

    def create_lags(self, data):
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
        max_lag= 2  #biggest lag
        #if less obs than max lag +1 -> cannot create lagged features-> raise error
        if n_obs <= max_lag:
            raise ValueError(f"Data has {n_obs} rows, but Lag 12 requires at least 13 observations.")
        #initialize list to store lagged features
        X_list= []
        #identify which cols are targets vs features for prior
        self.feature_type_map =[] 
        #loop through lags and create lagged features
        for lag in self.lag_indices:
            start= max_lag -lag  #start point for this lag
            end =n_obs- lag     #end point for this lag
            X_list.append(X_raw[start:end, :])  #append lagged features to list
            self.feature_type_map.extend([0]* self.n_vars)  #mark lagged features as type 0 for prior purposes
            X_list.append(Y_raw[start:end, :])     #append lagged targets
            self.feature_type_map.extend([1] * self.n_exog)
        #combine lagged features
        X_combined=np.column_stack(X_list)
        #want to deduplicate if lags of different variables are the same
        df_temp =pd.DataFrame(X_combined) 
        is_duplicate= df_temp.T.duplicated().values  #boolean mask for duplicate columns
        self.kept_indices= np.where(~is_duplicate)[0]   #store indices of unique features
        #keep only unique features
        X_unique =X_combined[:, self.kept_indices]
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
        L= len(self.lag_indices)       
        #prior mean: identity for first own-lag, zeros otherwise
        Phi_0= np.zeros((K, N))         
        #prior tightness matrix
        psi_diag =np.zeros(K)        
        #intercept
        psi_diag[0]= a3  
        #loop through lag blocks and variables to set tightness and prior mean       
        for j in range(len(sigmas_x)):
            #column index in the full Phi matrix (including intercept->+1)
            col_idx= j+1             
            n_preds_raw= len(sigmas_x)//len(self.lag_indices) #number of predictors before deduplication
            lag_idx = j//n_preds_raw     #which lag block this feature belongs to (0 for lag 0, 1 for lag 1, etc.)
            k= max(self.lag_indices[lag_idx], 1)  #scaling factor based on lag position (treat lag 0 as 1 for scaling to avoid div by zero)
            
            #use the specific sigma for this feature
            s_jj= sigmas_x[j]            
            #prior variance for this feature based on Minnesota-style logic
            psi_diag[col_idx]=(lam**2) /((k**2)*(s_jj**2))
        return Phi_0, np.diag(psi_diag)


    def fit(self, data,n_draws=2000, burn_in=500):
        """estimate bvar
        """
        #create lagged matrices calling fct
        X, Y= self.create_lags(data)
        #get dimensions
        T, N=Y.shape    #nr of equations and obs.
        K=self.n_features  #nr of features

        #calc res variances from univariate AR
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
        L= len(self.lag_indices)  #number of lag blocks
        sigmas_x_tiled=np.tile(sigmas_x_raw, L)        #tile the raw sigmas according to the lag structure
        
        # Apply the same deduplication mask used in create_lags
        sigmas_x = sigmas_x_tiled[self.kept_indices]
        #Minnesota config
        #-----------------
        if 'minnesota' in self.prior_type:
            #def default params
            a3= self.params.get('alpha', 100.0)
            #initial guesses for log(a1) and log(a2)
            init_params=[np.log(0.01), np.log(0.15)] 
            #residuals by optimizing the log marginal likelihood with respect to a1 and a2 
            res =optimize.minimize(self.log_ml_minnesota, init_params, args=(X, Y, a3, sigmas, sigmas_x),bounds=[(np.log(0.01), np.log(0.3)), (np.log(0.01), np.log(0.5))])
            #get optimal a1 and a2 by exponentiating the log values
            #a1, a2=np.exp(res.x)
            a1=0.05
            a2=0.025
            #create dummy observations based on final lambda
            X_dum, Y_dum= self.minnesota_dummies(a1, a2,a3, sigmas, sigmas_x)
            
            
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
            mu= self.params.get('mu', 1.0)
            a3= self.params.get('alpha', 100.0)
            #want to find best lambda=a1=a2 by maximing log marginal likelihood
            objective= lambda log_lam: -self.log_ml_natural(log_lam, X, Y, a3, sigmas, sigmas_x)  #objective to minimize (negative log marginal likelihood)
            res=optimize.minimize_scalar(objective, bounds=(-2.0, 0.0), method='bounded')      #search for best lambda in log-space
            best_lambda= np.exp(res.x)  #optimal lambda
            #set a1 and a2 to  the same optimized value
            a1= best_lambda
            a2= best_lambda       

            #posterior with dummies used for posterior estimation
            #get dummy obs
            X_dum, Y_dum= self.minnesota_dummies(a1, a2,a3, sigmas, sigmas_x)
            #augment data: instead of making giant cov matrix-> add rows to data
            Y_star=np.vstack([Y_dum, Y])
            X_star=np.vstack([X_dum, X])
            #add jitter to diagonal to avoid multicolinearity proble
            XX= X_star.T @ X_star
            jitter= np.eye(K)*1e-7   #tiny nudge to make positive definite
            Psi_post_inv =XX+jitter  #posterior precision
            #posterior mean
            XY=X_star.T @Y_star
            Phi_post=np.linalg.solve(Psi_post_inv, XY)
            #calc posterior covariance
            Psi_post =np.linalg.inv(Psi_post_inv)
            #posterior scale matrix
            residuals= Y_star -X_star @Phi_post
            S_post= residuals.T @residuals
            #posterior degrees of freedom
            nu_post= N+2+T
            #direct sampling: draw from Inverse-Wishart
            self.sigma_draws =stats.invwishart.rvs(df=nu_post, scale=S_post,size=n_draws)
            if n_draws== 1:
                self.sigma_draws = np.expand_dims(self.sigma_draws, axis=0)
            #initialize storage for phi draws
            self.phi_draws=np.zeros((n_draws, K, N))
            #cholesky decomp of coefficient covariance matrix
            L_Psi =np.linalg.cholesky(Psi_post)
            #iterate to draw phi coefficients
            for draw in range(n_draws):
                Sigma=self.sigma_draws[draw] #draw sigma   
                #cholesky of sigma
                L_Sigma =np.linalg.cholesky(Sigma)  
                    
                #matrix-normal draw
                Z=np.random.normal(size=(K, N))
                self.phi_draws[draw]= Phi_post+L_Psi @Z@L_Sigma.T
            
        return self
    
    def shapley_params(self, data, target_idx):  # <--- Make sure target_idx is here
        """
        Helper to extract inputs needed for Shapley value calculation.
        """
        # 1. Recreate the specific input vector x_t used for forecasting
        # Extract feature columns (excluding targets)
        feature_cols = [c for c in data.columns if 'target_' not in c]
        X_raw = data[feature_cols].values
        n_obs = X_raw.shape[0]
        
        # Extract lags
        lags = []
        for lag in self.lag_indices:
            idx = n_obs - lag - 1
            lags.append(X_raw[idx, :])
            
        X_combined = np.concatenate(lags)
        
        # Apply the exact same deduplication mask derived during training
        X_unique = X_combined[self.kept_indices]
        
        # 2. Create Feature Names (so we know which lag is which)
        raw_names = feature_cols
        all_names = []
        for lag in self.lag_indices:
            for name in raw_names:
                all_names.append(f"{name}_lag{lag}")
        
        # Filter names using the kept_indices
        final_names = [all_names[i] for i in self.kept_indices]
        
        # 3. Get Mean Coefficients (Posterior Mean)
        # phi_draws shape is (n_draws, K, N_vars). Take mean across draws.
        beta_mean_all = np.mean(self.phi_draws, axis=0) # Shape (K, N_vars)
        
        # Select coefficients for THIS specific target variable
        beta_target = beta_mean_all[:, target_idx]
        
        # Split Intercept (index 0) from Coefficients (indices 1:)
        # (Assuming the first column of X constructed in BVAR.fit was the intercept)
        intercept = beta_target[0]
        coeffs = beta_target[1:]
        
        # 4. Package for the utility function
        # Create a Series so the shap_values function can match names
        x_series = pd.Series(X_unique, index=final_names)
        coeffs_dict = dict(zip(final_names, coeffs))
        
        return x_series, coeffs_dict, intercept
    
    def forecast(self, data):
        #extract only the predictor columns for create lags logic
        feature_cols= [c for c in data.columns if 'target_' not in c]
        X_raw= data[feature_cols].values
        #initialize list to store lagged values
        lags= []
        n_obs= X_raw.shape[0] #number of observations in the input data
        #extract current values and lags
        for lag in self.lag_indices:
            idx= n_obs- lag-1   #index to extract the correct lagged value
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