import numpy as np
from scipy import stats, optimize, special

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
        


    def create_lags(self, data, horizon=1):
        """create lagged data matrices X (Tx(Np+1)) and Y matrix (TxN) for VAR"""
        #define Y
        Y = data.values
        T, N = Y.shape  #get dimensions
        self.n_vars= N  #store number of variables
        #create lag matrix
        X_list= []  #to store lagged values
        #create lags based on t, t-1,... 
        for lag in range(1, self.p+1):
            X_list.append(Y[self.p -lag:-lag, :])  #append lagged values: from T-p-lag to T-lag
        
        #concatenate lagged values
        X_lags= np.column_stack(X_list)  
        #add intercept column
        X =np.column_stack([np.ones(X_lags.shape[0]), X_lags])
        #store number of features 
        self.n_features= X.shape[1]  
        #cut Y so that if horizon is 1, we predict Y_{t+1} using X_t
        Y_cut= Y[self.p +horizon- 1: , :]
        #X and Y same length-> slice X to end where Y ends
        if horizon >1:
            X_cut= X[:-(horizon -1), :]
        else:
            X_cut= X

        return X_cut, Y_cut     #return matrix Xand Y
    

    def minnesota_dummies(self, N, a1, a3, sigmas): 
        """creates dummy observations: minnesota beliefs into data rows"""
        p= self.p  #number of lags
        #initialize prior dummy matrices
        yd_1=np.zeros((N*p,N))
        xd_1=np.zeros((N*p, 1+ N*p))
        #loop through variables: lag of var i should be 0 or 1 
        for lag in range(1, p+1):
            for i in range(N):
                row= (lag -1)*N +i  #row index
                #minnesota belief: distant lags have smaller variance
                scaling= (lag**1)/a1
                #get lag we put prior on
                col_idx= 1 +(lag-1)*N +i
                xd_1[row, col_idx]= scaling*sigmas[i]  
                #minnesota belief: own first lag=RW, else WN-> leave at 0
                if lag==1:
                    yd_1[row, i]= scaling*sigmas[i]
        #intercept prior
        yd_2=np.zeros((1, N))
        xd_2=np.zeros((1, 1+ N*p))
        xd_2[0,0]= 1.0/a3  #inverse variance
        #covariance prior
        yd_3=np.zeros((N, N))
        xd_3=np.zeros((N, 1+ N*p))
        np.fill_diagonal(yd_3, sigmas)  #prior is based on univariate ARs

        X_dum= np.vstack([xd_1, xd_2, xd_3])  #stack up dummy of predictos
        Y_dum= np.vstack([yd_1, yd_2, yd_3])   #stack up dummy of targets
        return X_dum, Y_dum



    def natural_moments(self, N, lam, a3, sigmas):
        """construct moments for natural niw (with minnesota function would have dim problems)
        """
        #nr of features per equation
        K= 1+ N*self.p
        #prior means (as before)
        Phi_0=np.zeros((K, N))
        for idx in range(N):
            Phi_0[1+idx, idx]=1.0   #Random walk assumption
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


    def fit(self, data, horizon=1):
        """estimate bvar
        """
        #create lagged matrices calling fct
        X, Y= self.create_lags(data, horizon=horizon)
        #get dimensions
        T, N=Y.shape    #nr of equations and obs.
        K=self.n_features  #nr of features

        #calc res variances from univariate AR
        sigmas=np.zeros(N)  #to store scales
        for idx in range(N):
            #get dependent variables for univariate AR
            y_i =Y[:, idx]
            #use own p lags-> calc residuals
            res=np.linalg.lstsq(X, y_i, rcond=None)[0]
            residuals= y_i-X @res
            sigmas[idx]= np.sqrt(np.sum(residuals**2)/ (T-K))  #store std dev of residuals

        #Minnesota config
        #-----------------
        if 'minnesota' in self.prior_type:
            #def intercept tightness to 100 by default->loose (is common choice)
            a3=self.params.get('alpha', 100.0)


            #glp optimization->best lambda
            res=optimize.minimize_scalar(lambda loglam: -self.log_ml_dummies(loglam, X, Y, a3, sigmas), bounds=(-5.0, 1.0), method='bounded')
            #optimal lambda
            final_lambda=np.exp(res.x)
            #create dummy observations based on final lambda
            X_dum, Y_dum= self.minnesota_dummies(N, final_lambda, a3, sigmas)
            #augment data: instead of making giant cov matrix-> add rows to data
            Y_star=np.vstack([Y_dum, Y])
            X_star=np.vstack([X_dum, X])
            #posterior estimation via OLS:
            XX_star= X_star.T @X_star  #X'X
            XX_inv= np.linalg.inv(XX_star)  #inverse
            Phi_post= XX_inv@(X_star.T @Y_star) #posterior mean
            #calc residuals
            residuals= Y_star -X_star @Phi_post
            #posterior scale matrix
            S_post= residuals.T @residuals
            #draws with gibbs sampling
            n_draws=2000
            df_post= Y_star.shape[0]- K  #posterior degrees of freedom

            self.Sigma_draws= np.zeros((n_draws, N, N))  #initialize storage for sigma draws
            self.Phi_draws= np.zeros((n_draws, K, N))  #initialize storage for phi draws
            #precompute cholesky of XX_inv
            L_XX=np.linalg.cholesky(XX_inv)
            #iterate to draw phi coefficients
            for draw in range(n_draws):
                #draw sigma through Inverse Wishart
                Sigma=stats.invwishart.rvs(df=df_post, scale=S_post)
                self.Sigma_draws[draw]=Sigma
                L_Sigma= np.linalg.cholesky(Sigma)  #Cholesky of sigma
                #standard normal draws to generate correlated normals
                Z= np.random.normal(size=(K, N))
                #draw phi
                self.Phi_draws[draw]= Phi_post + L_XX @Z @L_Sigma.T

            
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
            Phi_0, Psi_0= self.natural_moments(N, lam, theta,a3, sigmas)   #get mean and prior precision through fct
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
        """iterative system forecasting"""
        #get most recent lag window from data
        Y_hist=data.values 
        #create regressor vector for time T
        lags=[]
        for lag in range(1, self.p+1):
            lags.append(Y_hist[-lag, :])  #append lagged values
        #flatten and add intercept
        x_t =np.concatenate([[1.0], np.concatenate(lags)])
        N=self.n_vars       #nr of variables
        #determine how many B simulations
        n_draws=self.Phi_draws.shape[0]
        #initialize 0-array for predictions
        preds =np.zeros((n_draws, N))

        for idx in range(n_draws):
            #select coefficient matrix
            Phi = self.Phi_draws[idx]
            #if have unique Cov Matrix-> different one for each draw
            if self.Sigma_draws.ndim==3:
                Sigma =self.Sigma_draws[idx]
            else:
                Sigma=self.Sigma_draws  #same for each draw
            #direct prediction of Y_{T+h}
            pred=x_t@Phi+np.random.multivariate_normal(np.zeros(N), Sigma)
            preds[idx,:] =pred      #store prediction 

        #return predictions
        return preds
