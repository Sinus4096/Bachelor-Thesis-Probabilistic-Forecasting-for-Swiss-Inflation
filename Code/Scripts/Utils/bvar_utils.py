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

        #prior precision (inverse covariance) matrix
        Omega_0_inv= np.zeros((K, K))
        #intercept: loose prior (variance 1000^2-> precision 1e-6
        Omega_0_inv[0,0]= 1e-6
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
                    Omega_0_inv[idx, idx]= 1.0/sigma_sq
        return Phi_0, Omega_0_inv   #return prior mean and prior precision matrix
    
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


        #prior hyperparameters
        lam=self.params.get('lambda', 0.2)  #shrinkage parameter
        theta= self.params.get('theta', 0.5)    #overall tightness
        alpha_decay = self.params.get('alpha', 2.0)   #decay parameter
        #scale parameters from univariate AR residuals
        sigmas = self.ar_residual_scales(data)




        
    def log_marginal_likelihood(self, log_lambda, X, Y, XTX, XTY):
        """compute log marginal likelihood for given lambda"""
        #derive lambda as exponential of log_lambda to ensure positivity
        lambda_val= np.exp(log_lambda)
            #get prior precision matrix
        n_samples, n_features= X.shape       #number of features
        A_0= self.minnesota_precision(n_features, lambda_val)   #prior precision matrix
            
         #compute posterior parameters
        A_n= XTX +A_0         #posterior precision

        #compute posterior mean and variance
        A_n_inv= np.linalg.inv(A_n) #inverse
        log_det_An= 2*np.sum(np.log(np.diag(A_n_inv)))  #log determinant via Cholesky
        log_det_A0= np.sum(np.log(np.diag(A_0)))    #log determinant of prior precision

        beta_n= np.linalg.solve(A_n, XTY)    #posterior mean
        yy= np.dot(Y,Y)     #calculate the sum of squares of Y
        S_n= yy-np.dot(beta_n.T, np.dot(A_n, beta_n))  #sum of squares residuals
        #if S_n invalid, return negative infinity
        if S_n <=0:
            return -np.inf  
        return 0.5*(log_det_An- log_det_A0)- (n_samples/2.0)*np.log(S_n)    #return log marginal likelihood, calculated as per standard formula
        
    def fit(self, X, Y):
        """fit the hierarchical Bayesian regression model
    """
        #standardize features X
        self.X_mean= np.mean(X, axis=0)
        self.X_std= np.std(X, axis=0)
        self.X_std[self.X_std==0]=1.0     #avoid division by zero
        X_scaled= (X -self.X_mean)/ self.X_std  #standardized X

        #add intercept
        X_design= np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])  #add column of intercepts with 1s
        y_vec= Y.values  #convert to numpy array

        #precompute X'X and X'Y
        n_samples, n_features= X_design.shape #get dimensions
        XTX= X_design.T@X_design         #X'X
        XTY= X_design.T @y_vec   #X'Y

        #minnesota prior hyperparameter optimization
        if self.prior_type=='minnesota':
            final_lambda= self.params.get('lambda', 0.2)    #default lambda
            if self.shrinkage=='hierarchical':
                #optimize log marginal likelihood to find best lambda
                res= optimize.minimize_scalar(lambda log_lam: -self.log_marginal_likelihood(log_lam, X_design, y_vec, XTX, XTY),bounds=(-3.0, 1.0), method='bounded')
                final_lambda= np.exp(res.x)      #optimal lambda

            #posterior parameters
            A_0= self.minnesota_precision(n_features, final_lambda)   #prior precision matrix
            A_n= XTX +A_0   #posterior precision
            beta_hat= np.linalg.solve(A_n, XTY)  #posterior mean
            residuals= y_vec - X_design @beta_hat  #residuals estimation
            sse=np.sum(residuals**2)  #sum of squared errors
               
            #sampling from posterior
            n_draws= self.params.get('sampling_iters', 2000)  #number of posterior draws
            sig2_draws= stats.invgamma.rvs(a=(n_samples/2.0), scale=(sse/2.0), size=n_draws)  #draws of error variance
            self.sigma_draws=np.sqrt(sig2_draws)  #store std dev draws
            self.beta_draws= np.zeros((n_draws, n_features))  #initialize beta draws storage
            A_n_inv= np.linalg.inv(A_n)  #inverse of posterior precision
            #iterate to draw beta coefficients
            for draw in range(n_draws):
                cov_beta= A_n_inv*sig2_draws[draw]  #covariance of beta
                self.beta_draws[draw,:]= stats.random.multivariate_normal(mean=beta_hat, cov=cov_beta)  #draw beta coefficients randomly from posterior distribution and store
               
        #for independent normal-inverse wishart prior
        elif self.prior_type=='niw':
            #prior hyperparameters
            n_iter= self.params.get('sampling_iters', 2000)  #number of posterior draws
            burn_in= int(n_iter*0.2)    #burn-in period (=period to discard initial samples for convergence)
            beta_current=np.zeros(n_features)   #current beta is zero vector
            sigma_current=1.0  #initialize current error std dev
            lambda_current=0.2   
            self.beta_draws= np.zeros((n_iter-burn_in, n_features))     #initialize storage for beta draws
            self.sigma_draws= np.zeros(n_iter-burn_in)  #initialize storage for sigma draws
        
        
            #iterate
            for it in range(n_iter):
                #update beta given sigma2
                V0_inv=np.eye(n_features)*(1.0/(lambda_current**2))     #prior precision
                V0_inv[0,0]=1e-6   #no shrinkage on intercept
                V_post = np.linalg.inv(V0_inv + (XTX/sigma_current**2)) #posterior covariance
                mu_post= V_post @(XTY/sigma_current**2)   #posterior mean
                beta_current =np.random.multivariate_normal(mu_post, V_post)  #draw new beta

                #update sigma given beta
                residuals= y_vec-X_design @beta_current  #compute residuals
                ssr=np.dot(residuals,residuals)   #sum of squared residuals
                sigma_current =np.sqrt(stats.invgamma.rvs(a=(n_samples/2.0 +0.001), scale=(ssr/2.0+0.001)))  #draw new sigma by updating from inverse-gamma, add 0.001 to avoid issues

                #update lambda
                if self.shrinkage=='hierarchical':
                    #define grid for lambda to sample from
                    grid_lambda= np.linspace(0.01, 2.0, 50)
                    b_slopes= beta_current[1:]  #exclude intercept
                    k= len(b_slopes)      #number of slopes
                    log_post= []  #initialize store log posterior values
                    for l_val in grid_lambda:
                        #compute log posterior for each lambda in grid
                        ll=-k*np.log(l_val)- (np.sum(b_slopes**2)/(2.0*l_val**2))   #log likelihood part
                        lp= stats.gamma.logpdf(l_val, a=2.0, scale=0.1)   #log prior part (gamma prior with shape=2, scale=0.5)
                        #append total log posterior
                        log_post.append(ll+lp)
                    #convert to probabilities: subtract max for numerical stability
                    probs= np.exp(np.array(log_post)- np.max(log_post))  
                    #notmalize to get sum of 1
                    probs= probs/ np.sum(probs)
                    #sample new lambda from grid based on posterior probabilities
                    lambda_current= np.random.choice(grid_lambda, p=probs)

                #store draws after burn-in
                if it >= burn_in:
                    self.beta_draws[it-burn_in,:]= beta_current
                    self.sigma_draws[it-burn_in]= sigma_current
        
        return self
        
    def predict(self, X):
        """make predictions with fitted model"""
        X_scaled= (X -self.X_mean)/ self.X_std          #standardize X
        X_design= np.column_stack([np.ones(X_scaled.shape[0]), X_scaled]) #add intercept column of 1s


        n_preds= X_design.shape[0]      #number of predictions
        n_draws= self.beta_draws.shape[0]   #number of posterior draws
        preds= np.zeros((n_preds, n_draws))         #initialize predictions storage

        #iterate over posterior draws to make predictions
        for draw in range(n_draws):
            y_hat= X_design @self.beta_draws[draw]   #predicted mean
            #add noise based on sigma draw
            noise= np.random.normal(loc=0.0, scale=self.sigma_draws[draw], size=n_preds)
            #store prediction 
            preds[:, draw]= y_hat+noise
            
        return preds      #return matrix of predictions 

