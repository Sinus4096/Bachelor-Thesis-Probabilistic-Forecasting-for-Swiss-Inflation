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
        target_cols = [c for c in data.columns if 'target_' in c]
        feature_cols = [c for c in data.columns if 'target_' not in c]
        
        Y_raw = data[target_cols].values
        X_raw = data[feature_cols].values
        
        self.n_vars = Y_raw.shape[1]
        n_obs = len(data)
        
        # Define the structure: lags 0, 1, and 12
        self.lag_indices = [0, 1, 12]
        max_lag = 12
        
        if n_obs <= max_lag:
            raise ValueError(f"Data has {n_obs} rows, but Lag 12 requires at least 13 observations.")

        X_list = []
        # REMOVED: The extra loops. We only need to build the list once.
        for lag in self.lag_indices:
            start = max_lag - lag
            end = n_obs - lag
            X_list.append(X_raw[start:end, :])

        X_combined = np.column_stack(X_list)
        
        # Deduplication logic
        df_temp = pd.DataFrame(X_combined)
        is_duplicate = df_temp.T.duplicated().values
        self.kept_indices = np.where(~is_duplicate)[0]
        
        # FIX: Apply kept_indices to X_combined BEFORE adding the intercept
        X_unique = X_combined[:, self.kept_indices]
        
        # Add Intercept (Size will be: 1 + number of unique features)
        # Based on your error, this should result in size 28
        X = np.column_stack([np.ones(X_unique.shape[0]), X_unique])
        
        # Align Y: Y starts from max_lag to match the lag history
        Y_aligned = Y_raw[max_lag:, :]
        
        self.n_features = X.shape[1]
        return X, Y_aligned

    def minnesota_dummies(self, Y, a1, a2,a3, mu, sigmas): 
        """creates dummy observations: minnesota beliefs into data rows"""
        N = self.n_vars            # Number of target variables (i)
        K = self.n_features        # Total features (intercept + lags*vars)
        L = len(self.lag_indices)  # Number of lags
        
        # Calculate number of predictors (assuming X contains lags of all variables)
        n_preds = (K - 1) // L 
        
        # 1. Prior for Lags (Section 1 in your Equation)
        n_dum_lags = K - 1
        yd_1 = np.zeros((n_dum_lags, N))
        xd_1 = np.zeros((n_dum_lags, K))
        
        for j_feat in range(n_dum_lags):
            # Identify which lag (k) and which predictor variable (j)
            lag_pos = j_feat // n_preds
            var_j_idx = j_feat % n_preds
            
            k = self.lag_indices[lag_pos]
            k_scale = max(k, 1) # Treat lag 0 as 1 for scaling to avoid div by zero
            
            col_idx = j_feat + 1 # +1 because index 0 is intercept
            
            # We create a dummy for each target equation i
            # In a standard BVAR dummy setup, we simplify this to one block:
            if var_j_idx < N:
                # OWN LAGS (j == i): Variance = a1 / k^2 
                # Dummy X = k * sigma_jj / sqrt(a1)
                # Note: sigma_ii == sigma_jj here
                xd_1[j_feat, col_idx] = (k_scale * sigmas[var_j_idx]) / np.sqrt(a1)
            else:
                # OTHER LAGS (j != i): Variance = (a2 * sigma_ii^2) / (k^2 * sigma_jj^2)
                # Dummy X = (k * sigma_jj) / sqrt(a2)
                # This follows your formula: x = sigma_ii / sqrt(variance)
                # We use the average sigma or the corresponding sigma for scaling
                s_jj = sigmas[var_j_idx] if var_j_idx < len(sigmas) else np.mean(sigmas)
                xd_1[j_feat, col_idx] = (k_scale * s_jj) / np.sqrt(a2)

        # 2. Intercept Prior (Equation 3: a3 * sigma_ii^2)
        # Dummy X = sigma_ii / sqrt(a3 * sigma_ii^2) = 1 / sqrt(a3)
        xd_2 = np.zeros((1, K))
        xd_2[0, 0] = 1.0 / np.sqrt(a3)
        yd_2 = np.zeros((1, N))
        
        # 3. Sigma Prior (Standard Inverse Wishart scaling)
        # This matches the 'fixed Σ' logic in your text
        yd_sig = np.diag(sigmas)
        xd_sig = np.zeros((N, K))
        
        X_dum = np.vstack([xd_1, xd_2, xd_sig])
        Y_dum = np.vstack([yd_1, yd_2, yd_sig])
        
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

    
    def log_ml_dummies(self, loglambda, X, Y, a3, sigmas):
        """compute log marginal likelihood for using minnesota dummies
        """
        #define parameters
        lambda_val= np.exp(loglambda)  
        #get dummy obs
        X_dum, Y_dum= self.minnesota_dummies(Y.shape[1], lambda_val, a3, sigmas)
        #stack data and dummies
        X_star= np.vstack([X_dum, X])
        Y_star= np.vstack([Y_dum, Y])
        #compute posterior components
        XX_star=X_star.T @X_star
        T_star, K=X_star.shape   #get dimensions: T_star=obs+dummies, K=nr of preds
        T_real= Y.shape[0]  #nr of actual obs
        #Cholesky dcomp of XX_star for numerical stability
        L=np.linalg.cholesky(XX_star)
        log_det_xx=2*np.sum(np.log(np.diag(L)))  #log determinant of X'X
        #posterior mean
        Phi_post= np.linalg.solve(XX_star, X_star.T @Y_star)

        #residuals
        residuals= Y_star -X_star @Phi_post
        #posterior scale matrix
        S_post= residuals.T @residuals
        #compute log determinant of S_post
        sign, log_det_s= np.linalg.slogdet(S_post)
        #compute log marginal likelihood
        log_ml= 0.5*Y.shape[1]* (log_det_xx)-0.5*(T_star-K)*log_det_s
        return log_ml
 
    

    def log_ml_natural(self, log_lambda, X, Y, a3, sigmas):
        """compute log marginal likelihood for natural niw prior"""
        #define parameters
        lam= np.exp(log_lambda)
        T, N=Y.shape  #get dimensions
        #prior based on lambda
        Phi_0, Psi_0= self.natural_moments(N, lam, a3, sigmas)
        Psi_0_inv= np.diag(1.0/np.diag(Psi_0))  #inverse-> prior covariance of coeffs
        #posterior parameters
        XX= X.T @X
        Psi_post_inv= Psi_0_inv +XX  #posterior precision
        #determinants
        log_det_Psi_0_inv= np.sum(np.log(np.diag(Psi_0_inv)))
        #log determinant of posterior precision
        L_post_inv= np.linalg.cholesky(Psi_post_inv)
        log_det_Psi_post_inv= 2.0*np.sum(np.log(np.diag(L_post_inv)))
        #posterior mean
        Phi_post=np.linalg.solve(Psi_post_inv, (Psi_0_inv @Phi_0 +X.T @Y))
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
        term_cov=(N/2.0)*(log_det_Psi_0_inv -log_det_Psi_post_inv)
        term_scale= (nu_0/2.0)*log_det_S_0-(nu_post/2.0)*log_det_S_post
        const_gamma=special.multigammaln(nu_post/2.0, N)- special.multigammaln(nu_0/2.0, N)  #gamma function terms
        log_ml= term_cov +term_scale + const_gamma- (T*N/2.0)*np.log(np.pi)
        return log_ml


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
            res = np.diff(Y[:, idx])
            sigmas.append(np.std(res) if len(res)>0 else 0.1)
        sigmas = np.array(sigmas)
        
        #Minnesota config
        #-----------------
        if 'minnesota' in self.prior_type:
            #def default params
            a1=self.params.get('lambda', 0.01)
            a2= self.params.get('theta', 0.15)
            mu= self.params.get('mu', 1.0)
            a3= self.params.get('alpha', 100.0)
            #glp optimization->best lambda (optional)
            #res=optimize.minimize_scalar(lambda loglam: -self.log_ml_dummies(loglam, X, Y, a3, sigmas), bounds=(-5.0, 1.0), method='bounded')
            #optimal lambda
            #final_lambda=np.exp(res.x)
            #create dummy observations based on final lambda
            X_dum, Y_dum= self.minnesota_dummies(Y, a1, a2,a3, mu, sigmas)
            
            
            #augment data: instead of making giant cov matrix-> add rows to data
            Y_star=np.vstack([Y_dum, Y])
            X_star=np.vstack([X_dum, X])
            #add jitter to diagonal to avoid multicolinearity proble
            XX= X_star.T @ X_star
            jitter= np.eye(K)*1e-7   #tiny nudge to make positive definite
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
            # AR(1) residual variances for scaling
            sigmas = []
            for i in range(N):
                res = np.diff(Y[:, i])
                sigmas.append(np.std(res) if len(res) > 1 else 0.1)
            sigmas = np.array(sigmas)
            
            #prior moments
            alpha_0=np.zeros(N*K)  #all zeros for WN
            V_alpha_0= np.zeros(N*K) #to store prior precision diag
            for n in range(N): # For each equation (Target Variable)
                for k in range(K): # For each predictor feature
                    idx = n * K + k
                    
                    if k == 0: # Intercept
                        V_alpha_0[idx] = a3
                    else:
                        # Logic to identify 'Own Lag' vs 'Cross Lag'
                        # feature_idx is k-1 (ignoring intercept)
                        # var_j is the variable index (0 to N-1) this feature belongs to
                        var_j = (k - 1) % N 
                        
                        if var_j == n:
                            # OWN LAG (AR component)
                            V_alpha_0[idx] = a1**2
                        else:
                            # CROSS LAG (Variable J predicting Variable N)
                            V_alpha_0[idx] = (a1 * a2)**2
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
            a1=self.params.get('lambda', 1.0)
            a2= self.params.get('theta', 1.0)    #no cross-variable shrinkage for natural conjugate
            mu= self.params.get('mu', 1.0)
            a3= self.params.get('alpha', 100.0)
            #glp optimization->best lambda (optional)
            #hyperparameter optimization for lambda/a1
            #res=optimize.minimize_scalar(lambda loglam: -self.log_ml_natural(loglam, X, Y, a3, sigmas), bounds=(-5.0, 1.0), method='bounded')
            #a1=np.exp(res.x)  #optimal lambda
            #calc residual scaling from univariate ARs
            sigmas=[]  #to store scales
            for idx in range(N):
                #fit univariate AR to get residual std
                res=np.diff(Y[:, idx])
                sigmas.append(np.std(res) if len(res)>0 else 0.1)
            sigmas= np.array(sigmas)  #convert to array

            #posterior if dummies used for posterior estimation
            if self.implementation_type== 'dummies':

                #get dummy obs
                X_dum, Y_dum= self.minnesota_dummies(Y, a1, a2,a3, mu, sigmas)
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
            
            elif self.implementation_type== 'analytical':
                #priors; Psi is relative tightness matrix
                Phi_0, Psi_0=self.natural_moments(N, a1, a3, sigmas)
                Psi_0_inv=np.diag(1.0/np.diag(Psi_0))              
                #prior scale matrix
                S_0 =np.diag(sigmas**2)
                nu_0=N+ 2    #prior degrees of freedom
                #posteriors (derived analytically)
                XX=X.T @X
                XY= X.T @ Y
                YY=Y.T @ Y
                Psi_post_inv=Psi_0_inv +XX  #posterior precision
                Psi_post =np.linalg.solve(Psi_post_inv, (Psi_0_inv @ Phi_0 + XY))
                               
                #comp of scale matrix
                term_prior= Phi_0.T @ Psi_0_inv @Phi_0
                term_post= Phi_post.T@ Psi_post_inv @Phi_post
                #calc posterior scale matrix
                S_post= S_0 +YY +term_prior -term_post
                nu_post = nu_0+T

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
    
    def forecast(self, data):
        # 1. Extract only the predictor columns (match create_lags logic)
        feature_cols = [c for c in data.columns if 'target_' not in c]
        X_raw = data[feature_cols].values
        
        lags = []
        n_obs = X_raw.shape[0]
        
        # 2. Extract current values and lags 1 and 12
        # Data must contain rows [t, t-1, ... t-12] (13 total)
        for lag in self.lag_indices:
            idx = n_obs - lag - 1 
            lags.append(X_raw[idx, :])

        # Combine into a single horizontal vector
        X_combined = np.concatenate(lags)
        
        # 3. Apply the EXACT same deduplication used during training
        X_unique = X_combined[self.kept_indices]
        
        # 4. Add the intercept
        x_t = np.concatenate([[1.0], X_unique])
        
        # Now x_t size should be 28, matching phi_draws
        n_draws = self.phi_draws.shape[0]
        preds=np.zeros((n_draws, self.n_vars))
        
        for i in range(n_draws):
            noise = np.random.multivariate_normal(np.zeros(self.n_vars), self.sigma_draws[i])
            # Matrix multiplication will now succeed: (28) @ (28, n_vars)
            preds[i, :] = x_t @ self.phi_draws[i] + noise
            
        return preds