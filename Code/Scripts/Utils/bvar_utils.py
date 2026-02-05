import numpy as np
from scipy import stats, optimize, special
import pandas as pd

#see thesis for formulas
class BVAR:
    """
    implementation of Bayesian VAR with natural conjugate priors
    """
    def __init__(self, lags=2, prior_type='minnesota',prior_params=None):
        #initialize model
        self.p= lags  #number of lags
        self.prior_type =prior_type      #is minnesota by default
        self.params= prior_params if prior_params else {'lambda': 0.2, 'theta':0.5, 'a3': 100.0, 'alpha':2.0} #default prior params if none provided
        self.phi_draws= None        #to store posterior draws of coefficients
        self.sigma_draws= None  #to store posterior draws of error variances

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
        N= self.n_vars  #number of variables
        K = self.n_features  #total number of features
        # 1. Tightness around zero for most coefficients
        n_dum_1 = K - 1  # one dummy per feature (excluding intercept)
        yd_1 = np.zeros((n_dum_1, N))
        xd_1 = np.zeros((n_dum_1, K))
        
        # Add small regularization on all features
        for i in range(n_dum_1):
            xd_1[i, i+1] = a1  # regularize each feature (skip intercept at index 0)
        
        # 2. Intercept prior - shrink toward sample means
        y_bar = np.mean(Y[:min(10, len(Y)), :], axis=0)
        yd_2 = np.diag(y_bar) / mu
        xd_2 = np.zeros((N, K))
        xd_2[:, 0] = y_bar / mu  # prior on intercept
        
        # 3. Overall scale prior
        yd_3 = (y_bar / a3).reshape(1, -1)
        xd_3 = np.zeros((1, K))
        xd_3[0, 0] = 1.0 / a3
        
        # 4. Sigma Prior (Standard Inverse Wishart scale)
        yd_sig = np.diag(sigmas)
        xd_sig = np.zeros((N, K))
        
        X_dum = np.vstack([xd_1, xd_2, xd_3, xd_sig])
        Y_dum = np.vstack([yd_1, yd_2, yd_3, yd_sig])
        
        return X_dum, Y_dum


    def natural_moments(self, N, lam, a3, sigmas):
        """construct moments for natural niw (with minnesota function would have dim problems)
        """
        #nr of features per equation
        K= 1+ N*self.p
        #prior means (as before)
        Phi_0=np.zeros((K, N))
        for idx in range(N):
            Phi_0[1+idx, idx]=0.0   #White Noise assumption:  Swiss inflation often means reverts quickly
        #prior variance 
        Psi_diag=np.zeros(K)
        #intercept variance
        Psi_diag[0]=a3
        #loop through lags for lag variances
        for lag in range(1, self.p+1):
            #adapt variance logic for Kronecker structure
            scaling= (lam**2)/(lag**2)
            for j in range(N):
                #idx in the K-vector
                idx=1+((lag-1)*N)+j

                #scaling with sigmmas
                Psi_diag[idx]= scaling
        #convert to matrix
        Psi_0= np.diag(Psi_diag)
        return Phi_0, Psi_0
    
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
            a1 = self.params.get('lambda', 0.01)
            a2 = self.params.get('theta', 0.15)
            mu = self.params.get('mu', 1.0)
            a3 = self.params.get('alpha', 100.0)
            #glp optimization->best lambda
            #res=optimize.minimize_scalar(lambda loglam: -self.log_ml_dummies(loglam, X, Y, a3, sigmas), bounds=(-5.0, 1.0), method='bounded')
            #optimal lambda
            #final_lambda=np.exp(res.x)
            #create dummy observations based on final lambda
            X_dum, Y_dum= self.minnesota_dummies(Y, a1, a2,a3, mu, sigmas)
            
            
            #augment data: instead of making giant cov matrix-> add rows to data
            Y_star=np.vstack([Y_dum, Y])
            X_star=np.vstack([X_dum, X])
            # --- NUMERICAL STABILITY FIX ---
            # We add 'jitter' to the diagonal of X'X
            XX = X_star.T @ X_star
            jitter = np.eye(K) * 1e-7 # Tiny nudge to make it positive definite
            XX_stable = XX + jitter
            
            # Use solve instead of inv
            L_precision = np.linalg.cholesky(XX_stable)
            Phi_post = np.linalg.solve(XX_stable, X_star.T @ Y_star)
            
            phi_list, sigma_list = [], []
            Sigma = np.diag(sigmas**2)

            for i in range(n_draws + burn_in):
                # Draw Sigma
                res = Y_star - X_star @ Phi_post
                S_post = res.T @ res + np.eye(N) * 1e-6
                Sigma = stats.invwishart.rvs(df=T + N, scale=S_post)
                
                # Draw Phi
                Z = np.random.standard_normal((K, N))
                L_S = np.linalg.cholesky(Sigma)
                # Efficiently draw from Matrix Normal using Cholesky of precision
                innovation = np.linalg.solve(L_precision.T, Z) @ L_S.T
                Phi = Phi_post + innovation
                
                if i >= burn_in:
                    phi_list.append(Phi)
                    sigma_list.append(Sigma)

            self.phi_draws = np.array(phi_list)
            self.sigma_draws = np.array(sigma_list)

                
        #independent normal-inverse wishart prior
        #----------------------------------
        elif 'independent_niw' in self.prior_type:
            #prior hyperparameters
            lam=0.2    #overall tightness eq 6 (=a1)
            theta=0.5  #shrinkage on cross lags (=a2 in thsis)
            #def intercept tightness to 100 by default->loose (is common choice)
            a3=self.params.get('alpha', 100.0)
            n_iter =self.params.get('sampling', {}).get('n_draws', 2000)  #number of posterior draws
            burn_in= self.params.get('sampling', {}).get('burn_in', 500)  #burn-in period-> discard initial samples for convergence

            #prior moments
            Phi_0, Psi_0= self.natural_moments(N, lam, a3, sigmas)   #get mean and prior precision through fct
            V_alpha_0_inv = np.linalg.inv(Psi_0)  #prior precision matrix
            alpha_0= Phi_0.flatten(order='F')   #vectorize prior mean
            #prior scale matrix
            S_0=np.diag(sigmas**2)
            #prior degrees of freedom
            nu_0= N+2
            #initialize for Gibbs sampling
            Phi_current= Phi_0.copy()
            Sigma_current= np.diag(sigmas**2)   #start value for sigma
            #initialize storage for draws: vec bzw matrix of zeros
            self.Phi_draws= np.zeros((n_iter-burn_in, self.n_features, N))
            self.Sigma_draws= np.zeros((n_iter-burn_in, N, N))
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
                Phi_current=alpha_draw.reshape((self.n_features, N), order='F')

                #compute residuals
                residuals= Y -X @Phi_current
                #calc posterior scale matrix
                S_post= S_0+ residuals.T @residuals
                nu_post= nu_0+ T   #posterior degrees of freedom
                    
                #draw new Sigma from inverse-Wishart
                Sigma_current= stats.invwishart.rvs(df=nu_post, scale=S_post)

                #store draws after burn-in
                if it >= burn_in:
                    self.Phi_draws[it-burn_in]= Phi_current
                    self.Sigma_draws[it-burn_in]= Sigma_current
            

        # Natural Conjugate Normal-Wishart Prior
        # -------------------------------------
        elif 'natural_niw' in self.prior_type:
            #hyperparameters as before
            a3= self.params.get('alpha', 100.0)
            n_draws=self.params.get('sampling', {}).get('n_draws', 2000)

            #hyperparameter optimization for lambda/a1
            res=optimize.minimize_scalar(lambda loglam: -self.log_ml_natural(loglam, X, Y, a3, sigmas), bounds=(-5.0, 1.0), method='bounded')
            a1=np.exp(res.x)  #optimal lambda

            #priors; Psi is relative tightness matrix
            Phi_0, Psi_0=self.natural_moments(N, a1, a3, sigmas)
            Psi_0_inv=np.diag(1.0/np.diag(Psi_0))              
            #prior scale matrix
            S_0 =np.diag(sigmas**2)
            nu_0=N+ 2    #prior degrees of freedom
            #posteriors (derived analytically)
            XX=X.T @X
            Psi_post_inv=Psi_0_inv +XX  #posterior precision
            Psi_post =np.linalg.inv(Psi_post_inv)
                
            #post mean 
            Phi_post =Psi_post @(Psi_0_inv @Phi_0 +X.T @Y)                
            #efficient comp of scale matrix
            term_prior= Phi_0.T @ Psi_0_inv @Phi_0
            term_post= Phi_post.T@ Psi_post_inv @Phi_post
            term_data= Y.T @Y
            #calc posterior scale matrix
            S_post= S_0 +term_data +term_prior -term_post
            nu_post = nu_0+T

            #direct sampling: draw from Inverse-Wishart
            self.Sigma_draws =stats.invwishart.rvs(df=nu_post, scale=S_post, size=n_draws)
            #initialize storage for phi draws
            self.Phi_draws=np.zeros((n_draws, K, N))
            #cholesky decomp of coefficient covariance matrix
            L_Psi =np.linalg.cholesky(Psi_post)
            #iterate to draw phi coefficients
            for draw in range(n_draws):
                Sigma=self.Sigma_draws[draw] #draw sigma   
                #cholesky of sigma
                L_Sigma =np.linalg.cholesky(Sigma)  
                    
                #matrix-normal draw
                Z=np.random.normal(size=(K, N))
                self.Phi_draws[draw]= Phi_post+L_Psi @Z@L_Sigma.T
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
        preds = np.zeros((n_draws, self.n_vars))
        
        for i in range(n_draws):
            noise = np.random.multivariate_normal(np.zeros(self.n_vars), self.sigma_draws[i])
            # Matrix multiplication will now succeed: (28) @ (28, n_vars)
            preds[i, :] = x_t @ self.phi_draws[i] + noise
            
        return preds