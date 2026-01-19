import numpy as np
from scipy import stats, optimize
#see README for formulas
class BVAR:
    """
    implementation of Bayesian VAR with natural conjugate priors
    """
    def __init__(self, lags=2, prior_type='minnesota',prior_params=None):
        #initialize model
        self.p= lags  #number of lags
        self.prior_type =prior_type      #is minnesota by default
        self.params= prior_params if prior_params else {'lambda': 0.2, 'theta':0.5, 'alpha':2.0} #default prior params if none provided
        self.phi_draws= None        #to store posterior draws of coefficients
        self.sigma_draws= None  #to store posterior draws of error variances

        self.n_vars= None      #to store number of variables
        self.n_features= None  #to store number of features after lags (K= N*p +1)
        


    def create_lags(self, data):
        """create lagged data matrices X (Tx(Np+1)) and Y matrix (TxN) for VAR"""
        #define Y
        Y = data.values
        T, N = Y.shape  #get dimensions
        self.n_vars= N  #store number of variables
        #create lag matrix
        X_list= []  #to store lagged values
        for lag in range(1, self.p+1):
            X_list.append(Y[self.p-lag:-lag,:])  #append lagged values: from T-p-lag to T-lag
        
        #concatenate lagged values
        X_lags= np.column_stack(X_list)  
        #add intercept column
        X =np.column_stack([np.ones(X_lags.shape[0]), X_lags])
        #store number of features 
        self.n_features= X.shape[1]  
        #cut Y to align with lags
        Y_cut= Y[self.p:,:] 

        return X, Y_cut     #return matrix Xand Y

    def minnesota_moments(self, N, lam, theta, sigmas):
        """construct prior mean and variance for Minnesota prior, return mean vector and prior precision matrix
        """
        #number of features
        K=self.n_features
        #prior mean vector: zeros except for first lag of own variable
        Phi_0 = np.zeros((K, N))
        #iterate to set first lag own variable coefficients to 1
        for idx in range(N):
            Phi_0[1+idx, idx]=1.0

        #prior Variance initialization
        V_prior= np.zeros((K, K))
        #intercept: loose prior (variance 1000^2-> precision 1e-6
        V_prior[0,0]= 1e-6
        #iterate over variables and lags to set prior variances
        for lag in range(1, self.p+1):
            for i in range(N):
                for j in range(N):
                    #compute index in Phi
                    idx = 1+(lag-1)*N+ j
                    #lag decay (= 1/lag^2) (only calc lag^2 yet)
                    decay= (lag**2)
                    #own lag variance
                    if i==j:
                        sigma_sq= (lam/decay)**2
                    #cross lag variance
                    else:
                        scale_adj= (sigmas[i]/ sigmas[j])   #scale adjustment
                        sigma_sq= ((lam*theta)/decay)**2 *(scale_adj**2)
                    #set precision as inverse of variance
                    V_prior[idx, idx]= 1.0/sigma_sq
        return Phi_0, V_prior   #return prior mean and prior precision matrix
    
    def log_ml_glp(self, loglambda, X, Y, theta,sigmas):
        """compute log marginal likelihood for GLP optimization"""
        #define parameters
        lambda_val= np.exp(loglambda)  
        T, N=Y.shape  #get dimensions
        K= X.shape[1]  #number of features

        #get prior moments
        Phi_0, Omega_0_inv= self.minnesota_moments(N, lambda_val, theta, sigmas)
        #compute posterior parameters
        XX= X.T @X
        Omega_post_inv= Omega_0_inv + XX   #posterior precision
        #compute Cholesky decompositions for numerical stability
        L_prior=np.linalg.cholesky(Omega_0_inv+ np.eye(K)*1e-10)  #add small value for numerical stability
        L_post =np.linalg.cholesky(Omega_post_inv)
        #log determinants
        log_det_prior= 2.0*np.sum(np.log(np.diag(L_prior)))
        log_det_post= 2.0*np.sum(np.log(np.diag(L_post)))
        #compute post covariance and mean
        Omega_post= np.linalg.inv(Omega_post_inv) 
        Phi_post= Omega_post@ (X.T @Y + Omega_0_inv @Phi_0)

        #compute prior and posterior sum of squared errors
        S_0=np.diag(sigmas**2)   #prior scale matrix
        YTY= Y.T @Y  
        term1 =Phi_0.T @ Omega_0_inv @ Phi_0  #prior quadratic term
        term2= Phi_post.T @ Omega_post_inv@ Phi_post  #posterior quadratic term
        S_post=S_0+ YTY +term1- term2   #posterior scale matrix
        #compute log marginal likelihood
        log_ml= 0.5*(log_det_prior- log_det_post)-(T/2.0)*np.linalg.slogdet(S_post)[1]
        return log_ml   

    def fit(self, data):
        """estimate bvar
        """
        #create lagged matrices calling fct
        X, Y= self.create_lags(data)
        #get dimensions
        T, N=X.shape

        #calc univariate AR residuals for scaling 
        sigmas=np.zeros(N)  #to store scales
        for idx in range(N):
            #get dependent and independent variables for univariate AR
            y_i= Y[:, idx]
            x_i= X[:,1:2] #only own first lag

            #Minnesota config
            #-----------------
            if 'minnesota' in self.prior_type:
                #def prior theta(overall tightness)
                theta= self.params.get('theta',0.5)

                #check if hierarchical (estimate lambda) or fixed
                if isinstance(self.params.get('lambda'), dict):
                    #glp optimization
                    res=optimize.minimize_scalar(lambda loglam: -self.log_ml_glp(loglam, X, Y, theta, sigmas), bounds=(-3.0, 0.5), method='bounded')
                    #optimal lambda
                    final_lambda=np.exp(res.x)
                #else fixed lambda
                else:
                    final_lambda= self.params.get('lambda', 0.2)
                #get prior moments with fct
                Phi_0, Omega_0_inv= self.minnesota_moments(N, final_lambda, theta, sigmas)
                
                #calculate posterior parameters
                XX= X.T @X  
                Omega_post_inv= Omega_0_inv + XX   #posterior precision
                Omega_post= np.linalg.inv(Omega_post_inv)  #posterior covariance
                Phi_post =Omega_post@(X.T @Y+ Omega_0_inv @Phi_0)   #posterior mean
                #calculate residuals
                S_0=np.diag(sigmas**2)   #prior scale matrix
                S_post=S_0+Y.T @Y +Phi_0.T @ Omega_0_inv @Phi_0 - Phi_post.T@ Omega_post_inv @Phi_post  #posterior scale matrix
                
                #draws from posterior
                n_draws=2000
                nu_post=T+N+2 #posterior degrees of freedom
                self.Sigma_draws=stats.invwishart.rvs(df=nu_post, scale=S_post, size=n_draws)  #draws of sigma
                self.Phi_draws= np.zeros((n_draws, self.n_features, N))  #initialize storage for phi draws
                L_Omega = np.linalg.cholesky(Omega_post)  #Cholesky of posterior precision

                #iterate to draw phi coefficients
                for draw in range(n_draws):
                    #if run multiple sims, draw separate sigma 
                    if n_draws>1:
                        Sigma =self.Sigma_draws[draw]
                    #if just one sim, only one sigma
                    else:
                        Sigma= self.Sigma_draws
                    L_Sigma= np.linalg.cholesky(Sigma)  #Cholesky of sigma
                    #standard normal draws to generate correlated normals
                    Z= np.random.normal(size=(self.n_features, N))
                    #draw phi
                    self.Phi_draws[draw]= Phi_post + L_Omega.T @Z @L_Sigma.T
            
            #independent normal-inverse wishart prior
            #----------------------------------
            elif 'niw' in self.prior_type:
                #prior hyperparameters
                lam=0.2    #prior std dev of coefficients, fixed for simplicity
                theta=0.5  #shrinkage on cross lags
                n_iter =self.params.get('sampling', {}).get('n_draws', 2000)  #number of posterior draws
                burn_in= self.params.get('sampling', {}).get('burn_in', 500)  #burn-in period-> discard initial samples for convergence

                #prior moments
                Phi_0, Omega_0_inv= self.minnesota_moments(N, lam, theta, sigmas)   #get mean and prior precision through fct
                Phi_prior=Phi_0 #make copy of prior mean
                V_prior_inv= Omega_0_inv  #copy prior precision
                #initialize phi and sigma
                Phi_current= np.zeros((self.n_features, N))
                Sigma_current= np.eye(N)
                #initialize storage for draws: vec bzw matrix of zeros
                self.Phi_draws= np.zeros((n_iter-burn_in, self.n_features, N))
                self.Sigma_draws= np.zeros((n_iter-burn_in, N, N))

                XX= X.T @X
                XY=X.T @Y

                #iterate through draws to sample from posterior
                for it in range(n_iter):
                    #update Phi given Sigma
                    Sigma_inv= np.linalg.inv(Sigma_current)    #inverse of current sigma
                    V_post=np.linalg.inv(V_prior_inv +XX)  #posterior covariance
                    Phi_hat= V_post@(V_prior_inv @Phi_prior + XY)   #posterior mean estimate
                    #draw new Phi
                    L_V=np.linalg.cholesky(V_post)     #Cholesky of posterior covariance
                    L_S =np.linalg.cholesky(Sigma_current)   #Cholesky of current sigma
                    Z=np.random.normal(size=(self.n_features, N))   #standard normal draws
                    Phi_current= Phi_hat + L_V@Z @L_S.T    #draw new Phi

                    #update Sigma given Phi
                    residuals= Y - X @Phi_current   #compute residuals
                    S_post =np.dot(residuals.T, residuals) #posterior scale matrix
                    nu_post= T +N +2   #posterior degrees of freedom
                    #draw new Sigma from inverse-Wishart
                    Sigma_current= stats.invwishart.rvs(df=nu_post, scale=S_post)

                    #store draws after burn-in
                    if it >= burn_in:
                        self.Phi_draws[it-burn_in]= Phi_current
                        self.Sigma_draws[it-burn_in]= Sigma_current
        return self
    
    def forecast(self, data, horizon=12):
        """iterative system forecasting"""
        #get historical data as array
        Y_hist=data.values 
        #slice last p rows-> get starting lag window
        current_window=Y_hist[-self.p:,:] 
        N=self.n_vars       #nr of variables
        #determine how many B simulations
        n_draws=self.Phi_draws.shape[0]
        #initialize 0-array
        paths =np.zeros((n_draws, horizon, N))

        for idx in range(n_draws):
            #select coefficient matrix
            Phi = self.Phi_draws[idx]
            #if have unique Cov Matrix-> different one for each draw
            if self.Sigma_draws.ndim==3:
                Sigma =self.Sigma_draws[idx]
            else:
                Sigma=self.Sigma_draws  #same for each draw

            #copy to append new predictions iteratively
            temp_hist=current_window.copy()
            #iterate through all forecast horizons
            for h in range(horizon):
                lags =[]    #initialize list
                #iterate through nr of lags-> collect observations 
                for lag in range(1, self.p+1):
                    lags.append(temp_hist[-lag,:])
                #regressor vector, intecerpt=1.0
                x_t =np.concatenate([[1.0], np.concatenate(lags)])

                #next step prediction
                pred=x_t@Phi+np.random.multivariate_normal(np.zeros(N), Sigma)
                paths[idx,h,:] =pred      #store prediction 
                temp_hist =np.vstack([temp_hist, pred]) #append prediction

        #return collection of simulated paths
        return paths
