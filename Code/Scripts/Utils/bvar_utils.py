import numpy as np
from scipy import stats, optimize
#see README for formulas
class HierarchicalBayesianRegression:
    """
    implementation of Bayesian LR with Hierarchical Priors
    """
    def __init__(self, prior_type='minnesota', shrinkage='hierarchical', params=None):
        #initialize model
        self.prior_type=prior_type
        self.shrinkage=shrinkage
        self.params=params if params is not None else {} #default empty dict if no params provided
        self.beta_draws = None  #to store posterior draws of coefficients
        self.sigma_draws = None  #to store posterior draws of error variances
        self.X_mean= None       #to store means of X for prediction
        self.X_std= None  #to store std dev of X for prediction

        def minnesota_precision(self, n_features, lambda_param):
            """compute prior precision matrix for minnesota prior"""
            #prior precision= (1/lambd^2)*I
            precision= np.eye(n_features)*(1.0/(lambda_param**2))
            #on intercept no shrinkage
            precision[0,0]= 1e-6
            return precision
        
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

