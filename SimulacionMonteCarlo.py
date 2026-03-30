from abc import ABC, abstractmethod
import numpy as np
import scipy

class SimulacionMonteCarlo(ABC):

    @abstractmethod
    def informar_parametros(self):
        """
        Informa parámetros del modelo.
        """
        pass
        
    @abstractmethod
    def simular(self):
        """
        Devuelve una serie temporal simulada con los parámetros informados.
        """
        pass

    def dibuja_y_simula(self):
        return self.simular()


class GBM(SimulacionMonteCarlo):

    def __init__(self, param_dict=None):
        self.mu = None
        self.sigma = None
        self.S0 = None
        self.Deltat = None
        self.T = None
        self.N = None
        self.M = None
        self.param_dict = param_dict
        if param_dict is not None:
            self.informar_parametros(param_dict)


    def confirmar_parametros_validos(self, param_dict):
        todo_bien = True
        nones = sum([var not in param_dict.keys() for var in ('Deltat', 'N', 'T')])
        if  nones >= 2:
            print("ERROR: Hay que informar como mínimo dos de los tres siguientes parámetros:")
            print("Deltat, T, N")
            todo_bien = False
        elif nones == 1:
            if 'Deltat' not in param_dict.keys():
                param_dict['Deltat'] = param_dict['T'] / param_dict['N']
            elif 'T' not in param_dict.keys():
                param_dict['T'] = param_dict['Deltat'] * param_dict['N']
            elif 'N' not in param_dict.keys():
                param_dict['N'] = round(param_dict['T'] / param_dict['Deltat'])
        elif nones == 0:
            if param_dict['T']/param_dict['N'] != param_dict['Deltat']:
                print("ERRROR: Los valores de los siguientes parámetros no son válidos:")
                print("Deltat, T, N")
                todo_bien = False
        for var in ['mu', 'sigma', 'S0', 'M']:
            if var not in param_dict.keys():
                print(f"ERROR: {var} debe estar informada.")
                todo_bien = False
            elif var != 'mu' and param_dict[var] < 0:
                print(f"ERROR: {var} debe ser mayor o igual que cero.")
                todo_bien = False
        if param_dict['M'] == 0:
            print("WARNING: el número de simulaciones es cero.")
            todo_bien = False
        if param_dict['S0'] <= 0:
            print("ERROR: el precio inicial debe ser positivo.")
            todo_bien = False

        return todo_bien

    
    def informar_parametros(self, param_dict):        
        if self.confirmar_parametros_validos(param_dict):
            self.mu = param_dict['mu']
            self.sigma = param_dict['sigma']
            self.S0 = param_dict['S0']
            self.Deltat = param_dict['Deltat']
            self.T = param_dict['T']
            self.N = param_dict['N']
            self.M = param_dict['M']
            self.param_dict = param_dict
        else:
            print("ERROR: No se han asignado los parámetros.")
        

    def simular(self, S0=None, N=None, M=None):
        new_param_dict = self.param_dict.copy()
        if S0 is not None:
            new_param_dict['S0'] = S0
        if N is not None:
            new_param_dict['N'] = N
            new_param_dict['T'] = N * new_param_dict['Deltat']
        if M is not None:
            new_param_dict['M'] = M
        
        if not self.confirmar_parametros_validos(new_param_dict):
            print("ERROR: parámetros inválidos. Abortando simulación.")
            return
        else:
            self.informar_parametros(new_param_dict)
        
        S = np.full([self.N+1, self.M], self.S0, dtype=np.dtype(float))
        Z = np.random.standard_normal([self.N, self.M])    
        
        for i in range(self.N):
            S[i+1,:] = S[i,:]*(1 + self.mu * self.Deltat + self.sigma * np.sqrt(self.Deltat)*Z[i, :])

        return S
        

    def mostrar_parametros(self):
        print("===== Parámetros del modelo =====")
        for var in self.param_dict.keys():
            k = len(var)//8 + 2
            if var == 'sigma_J':
                k -= 1
            espacio = k*"\t"
            print(f"{var}:{espacio}{self.param_dict[var]}")
        
        


    def ajustar_parametros(self, ts, M = 1):
        # ajusta todos los parámetros o ninguno
        new_params = {'T': ts.shape[0], 'N': ts.shape[0], 'Deltat':1., 'S0':ts[0], 'M':M}
        def neg_log_likelihood(p):
            var_mu, var_sigma = p
            nlL = -np.sum([np.log(1/(np.sqrt(2*np.pi*new_params['Deltat'])*var_sigma))
                            -(ts[k]-ts[k-1]*(1+var_mu*new_params['Deltat']))**2/
                             (2*var_sigma**2*new_params['Deltat']*ts[k-1]**2)
                           for k in range(1, ts.shape[0])])
            return nlL

        opt_res = scipy.optimize.minimize(neg_log_likelihood, [1., 1.], method='nelder-mead',
                                          options={'xatol': 1e-8, 'disp': False}, )

        if opt_res.success:
            new_params['mu'] = opt_res.x[0]
            new_params['sigma'] = opt_res.x[1]
            self.informar_parametros(new_params)
            print("Ajuste de parámetros realizado correctamente.")
        else:
            print("ERROR: no se han ajustado los parámetros correctamente.")

        return


class MertonJumpDiffusion(GBM):
    def __init__(self, param_dict=None):
        super().__init__(param_dict)
        self.lbda = None
        self.mu_J = None
        self.sigma_J = None
        if param_dict is not None:
            self.informar_parametros(param_dict)
        

    def confirmar_parametros_validos(self, param_dict):
        todo_bien = super().confirmar_parametros_validos(param_dict)
        for var in ['lambda', 'sigma_J', 'mu_J']:
            if var not in param_dict.keys():
                print(f"ERROR: la variable {var} debe estar informada.")
                todo_bien = False
            if var != 'mu_J' and param_dict[var] <= 0:
                print(f"ERROR: la variable {var} debe ser positiva.")
                todo_bien = False
                
        return todo_bien

    
    def informar_parametros(self, param_dict):
        if not self.confirmar_parametros_validos(param_dict):
            return
        super().informar_parametros(param_dict)
        self.lbda = param_dict['lambda']
        self.mu_J = param_dict['mu_J']
        self.sigma_J = param_dict['sigma_J']

    def mostrar_parametros(self):
        super().mostrar_parametros()
        pass

    def simular(self, S0=None, N=None, M=None):
        S = super().simular(S0, N, M)

        N = np.random.poisson(size=[self.N+1, self.M], lam=self.lbda*self.Deltat)
        jumps = np.argwhere(N > 0)
        J = N * np.random.normal(size=N.shape, loc=self.mu_J, scale=self.sigma_J)
        S += J.cumsum(axis=0)
        for j in range(self.M):
            for i in range(self.N):
                if S[i,j] <= 0:
                    S[i:,j] = 0
                    break
        return S
        


    def ajustar_parametros(self, ts, M):
        new_params = {'T': ts.shape[0], 'N': ts.shape[0], 'Deltat':1., 'S0':ts[0], 'M':M}
        
        def neg_log_likelihood(Theta):
            k_MAX = 10
            
            mu, sigma = Theta[0], Theta[1]
            lbda = Theta[2]
            mu_J, sigma_J = Theta[3], Theta[4]

            log_likelihood = 0.
            for n in range(1, new_params['N']):
                ksum = 0.
                for k in range(k_MAX+1):
                    loc = ts[n-1]*(1+mu*new_params['Deltat']) + k*mu_J
                    scale = np.sqrt(ts[n-1]**2*new_params['Deltat']*sigma**2+k*sigma_J**2)
                    density_norm = scipy.stats.norm.pdf(x=ts[n],loc=loc,scale=scale)
                    mass_poisson = scipy.stats.poisson.pmf(k=k, mu=lbda)
                    ksum += density_norm*mass_poisson
                log_likelihood += np.log(ksum)
            return -log_likelihood

        def merton_nll(v, K_max=30):
            x = np.diff(np.log(ts))
            dt = 1.
            
            mu0       = v[0]
            sigma     = np.exp(v[1])
            lam       = np.exp(v[2])
            mu_j      = v[3]
            sigma_j   = np.exp(v[4])
        
            ks        = np.arange(K_max + 1)
            means     = mu0 * dt + ks * mu_j
            vars_     = sigma**2 * dt + ks * sigma_j**2
            log_pois  = scipy.stats.poisson.logpmf(ks, lam * dt)
            log_gauss = scipy.stats.norm.logpdf(x[:, None], means, np.sqrt(vars_))
            log_w     = log_pois[None, :] + log_gauss
        
            # log-sum-exp
            a   = log_w.max(axis=1, keepdims=True)
            ll  = (a.squeeze() + np.log(np.exp(log_w - a).sum(axis=1))).sum()
            return -ll

        def EM():
            def safe_em_step(S, dt, Theta, K_max=40):
                x   = np.log(ts[1:] / ts[:-1])
                N   = len(x)
                mu0 = Theta[0]
                sigma, sigma_j = Theta[1], Theta[4]
                lam, mu_j      = Theta[2], Theta[3]
            
                ks = np.arange(K_max + 1)
            
                # --- E-step (log space) ---
                means     = mu0 * dt + ks * mu_j                      # (K+1,)
                vars_     = sigma**2 * dt + ks * sigma_j**2           # (K+1,)
                
                # Guard: variance must be positive
                vars_     = np.maximum(vars_, 1e-12)
            
                log_gauss   = scipy.stats.norm.logpdf(x[:, None], means, np.sqrt(vars_))  # (N, K+1)
                log_poisson = scipy.stats.poisson.logpmf(ks, lam * dt)                    # (K+1,)
                log_w       = log_gauss + log_poisson[None, :]                 # (N, K+1)
            
                # log-sum-exp normalization
                a   = log_w.max(axis=1, keepdims=True)
                w   = np.exp(log_w - a)
                row_sums = w.sum(axis=1, keepdims=True)
            
                # Guard: if a row is all zeros something is very wrong
                if np.any(row_sums == 0):
                    raise ValueError("All weights zero for at least one observation. "
                                     "Check initialization and K_max.")
                w /= row_sums

                

                
                # --- Sufficient statistics ---
                A = w.sum()                          # = N
                B = (w * ks).sum()
                C = (w * ks**2).sum()
                D = (w * x[:, None]).sum()
                E = (w * ks * x[:, None]).sum()
            
                # --- M-step: joint mu0, mu_j ---
                det = A * C - B**2
                if abs(det) < 1e-10:
                    mu0_new  = D / (A * dt)
                    mu_j_new = mu_j                  # fallback: keep current
                else:
                    mu0_new  = (C * D - B * E) / (det * dt)
                    mu_j_new = (A * E - B * D) / det
            
                # --- M-step: lambda ---
                lam_new = np.clip((w * ks).sum() / (N * dt), 0.01, 200.0)
            
                # --- M-step: sigma_j ---
                resid    = x[:, None] - mu0_new * dt - ks * mu_j_new
                denom_j  = (w * ks**2).sum()
                sigma_j2 = (w * ks * resid**2).sum() / denom_j if denom_j > 1e-10 else sigma_j**2
            
                # --- M-step: sigma ---
                sigma2 = ((w * resid**2).sum() - sigma_j2 * (w * ks).sum()) / (N * dt)
            
                # Clamp variances
                sigma2   = max(sigma2,   0.002)
                sigma_j2 = max(sigma_j2, 1e-8)
                lam_new = max(lam_new, 1.)
            
                # --- Drift recovery ---
                k_bar   = np.exp(mu_j_new + 0.5 * sigma_j2) - 1
                mu_new  = mu0_new + 0.5 * sigma2 + lam_new * k_bar
            
                # --- NaN guard ---
                #new_params = dict(mu=mu_new, mu0=mu0_new, sigma=np.sqrt(sigma2),
                #                  lam=lam_new, mu_j=mu_j_new, sigma_j=np.sqrt(sigma_j2))
                Theta_new = [mu_new, np.sqrt(sigma2), lam_new, mu_j_new, np.sqrt(sigma_j2)]
            
                return Theta_new
            


            return Theta

        r = np.diff(np.log(ts))
        mu = np.mean(r)
        sigma = 0.05971123130259934
        lbda = 0.001
        mu_J = 50.
        sigma_J = 200.
        Theta = np.array([mu, sigma, lbda, mu_J, sigma_J])
        
        opt_res = scipy.optimize.minimize(merton_nll, Theta,  method='L-BFGS-B')
        Theta = opt_res.x
        new_params['mu'] = Theta[0]
        new_params['sigma'] = np.exp(Theta[1])
        new_params['lambda'] = np.exp(Theta[2])
        new_params['mu_J'] = Theta[3]
        new_params['sigma_J'] = np.exp(Theta[4])
        self.informar_parametros(new_params)    
        
        return 
            
                    
        

        










            
        