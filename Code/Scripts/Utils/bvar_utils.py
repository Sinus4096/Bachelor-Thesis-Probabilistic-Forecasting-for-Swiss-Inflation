import numpy as np
from scipy import stats, optimize
import scipy.linalg
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

    def minnesota_moments(self, N, a1, a2, a3, sigmas):
        """construct prior mean and variance for Minnesota prior (according to thesis notation)"""
        #number of features per equation
        K= 1+ N*self.p
        #total nr params in alpha
        M=N*K
        #prior mean vector: zeros except for first lag of own variable
        Phi_0 = np.zeros((K, N))
        #iterate to set first lag
        for idx in range(N):
            Phi_0[1+idx, idx]=1.0   #acc to thesis: own lags are 1 all remaining 0
        
        #vectorize to get alpha_bar
        alpha_0= Phi_0.T.flatten()  
        #prior Variance initialization
        V_prior= np.zeros(M)
        
        #loop through  equations
        for i in range(N):
            #get idx
            idx_int= i*K
            #intercept = a3*sigma_i^2
            V_prior[idx_int]= a3* (sigmas[i]**2)

            #loop through lags
            for lag in range(1, self.p+1):
                #loop through variables
                for j in range(N):
                    #calc index for coeff of var j at lag k in eq i
                    idx= (i*K)+1+((lag-1)*N)+j
                    #lag decay
                    decay= lag**2
                    #own lag variance
                    if i==j:
                        V_prior[idx]= a1/decay
                    #cross lag variance
                    else:
                        sigma_ii=sigmas[i]**2
                        sigma_jj=sigmas[j]**2
                        V_prior[idx]= (a2*sigma_ii)/(decay*sigma_jj)
        #set precision as diagonal matrix
        V_prior_cov= np.diag(V_prior)
        return Phi_0, V_prior_cov   #return prior mean and prior precision matrix
    
    def log_ml_glp(self, loglambda, X, Y, theta, a3, sigmas):
        """compute log marginal likelihood for GLP optimization"""
        #define parameters
        lambda_val= np.exp(loglambda)  
        T, N=Y.shape  #get dimensions
        K= X.shape[1]  #number of features

        #get prior moments
        Phi_0_full, V_prior= self.minnesota_moments(N, lambda_val, theta, a3, sigmas)
        #V_prior= MxM matrix for computation need KxK matrix
        Omega_0_inv= np.linalg.inv(V_prior[:K, :K])
        #also Phi_0 as KxN matrix
        Phi_0=Phi_0_full.reshape((K, N), order='F')
        #compute posterior parameters
        XX= X.T @X
        Omega_post_inv= Omega_0_inv + XX   #posterior precision
        #compute Cholesky decompositions for numerical stability
        L_prior=np.linalg.cholesky(Omega_0_inv+ np.eye(K)*1e-10)  #add small value for numerical stability
        L_post =np.linalg.cholesky(Omega_post_inv)
        #log determinants of precision matrixa
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
        log_ml =0.5 *(log_det_prior-log_det_post)-(T /2.0)*np.linalg.slogdet(S_post)[1]
        return log_ml   

    def fit(self, data):
        """estimate bvar
        """
        #create lagged matrices calling fct
        X, Y= self.create_lags(data)
        #get dimensions
        T, N=Y.shape    #nr of equations and obs.
        K=self.n_features  #nr of features

        #calc univariate AR residuals for scaling 
        sigmas=np.zeros(N)  #to store scales
        for idx in range(N):
            #get dependent variables for univariate AR
            y_i =Y[:, idx]
            #use own p lags-> calc residuals
            res=np.linalg.lstsq(X[:, 1:], y_i, rcond=None)[0]
            residuals= y_i - X[:, 1:] @res
            sigmas[idx]= np.sqrt(np.sum(residuals**2)/ (T-self.p-1))  #store std dev of residuals

            #Minnesota config
            #-----------------
            if 'minnesota' in self.prior_type:
                #def prior theta(overall tightness) 
                a2= self.params.get('theta',0.5)
                #def intercept tightness to 100 by default->loose (is common choice)
                a3=self.params.get('alpha', 100.0)

                #check if hierarchical (estimate lambda) or fixed
                if isinstance(self.params.get('lambda'), dict):
                    #glp optimization
                    res=optimize.minimize_scalar(lambda loglam: -self.log_ml_glp(loglam, X, Y, a2, a3, sigmas), bounds=(-3.0, 0.5), method='bounded')
                    #optimal lambda
                    final_lambda=np.exp(res.x)
                #else fixed lambda
                else:
                    final_lambda= self.params.get('lambda', 0.2)
                #get prior moments with fct
                Phi_0, V_prior= self.minnesota_moments(N, final_lambda, a2, a3, sigmas)
                
                #conver cov to precision 
                V_prior_inv=np.linalg.inv(V_prior)
                #calculate posterior parameters
                XX= X.T @X  
                V_post_inv= V_prior_inv + XX   #posterior precision
                V_post= np.linalg.inv(V_post_inv)  #posterior covariance
                Phi_post =V_post@(X.T @Y+ V_prior_inv @Phi_0)   #posterior mean
                #calculate scale matrix
                S_0=np.diag(sigmas**2)   #prior scale matrix
                S_post=S_0+Y.T @Y +Phi_0.T @ V_prior_inv@Phi_0 - Phi_post.T@ V_post_inv @Phi_post  #posterior scale matrix
                
                #draws from posterior
                n_draws=2000
                nu_post=T+N+2 #posterior degrees of freedom
                self.Sigma_draws=stats.invwishart.rvs(df=nu_post, scale=S_post, size=n_draws)  #draws of sigma
                self.Phi_draws= np.zeros((n_draws, self.n_features, N))  #initialize storage for phi draws
                L_V = np.linalg.cholesky(V_post)  #Cholesky of posterior precision

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
                    self.Phi_draws[draw]= Phi_post + L_V.T @Z @L_Sigma.T

            
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
                Phi_0, V_alpha_0_inv= self.minnesota_moments(N, lam, theta,a3, sigmas)   #get mean and prior precision through fct
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
