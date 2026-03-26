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
        for var in self.param_dict.keys():
            k = len(var)//8 + 2
            if var == 'sigma_J':
                k -= 1
            espacio = k*"\t"
            print(f"{var}:{espacio}{self.param_dict[var]}")
        
        


    def ajustar_parametros(self, ts, M = 1):
        # ajusta todos los parámetros o ninguno
        new_params = {'T': ts.shape[0], 'N': ts.shape[0], 'Deltat':1., 'S0':ts.iloc[0], 'M':M}
        def neg_log_likelihood(p):
            var_mu, var_sigma = p
            nlL = -np.sum([np.log(1/(np.sqrt(2*np.pi*new_params['Deltat'])*var_sigma))
                            -(ts.iloc[k]-ts.iloc[k-1]*(1+var_mu*new_params['Deltat']))**2/
                             (2*var_sigma**2*new_params['Deltat']*ts.iloc[k-1]**2)
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
        
        for j in range(self.M):
            for i in range(self.N+1):
                jump = N[i,j] * np.random.normal(loc=self.mu_J, scale=self.sigma_J)
                if S[i,j] + jump <= 0:
                    S[i:,j] = 0                    
                    break
                else:
                    S[i:,j] += jump
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

        def jac(Theta):
            jac = np.zeros(5)
            k_MAX = 20

            mu, sigma = Theta[0], Theta[1]
            lbda = Theta[2]
            mu_J, sigma_J = Theta[3], Theta[4]

            
            for n in range(1, new_params['N']):
                den = 0
                for k in range(k_MAX+1):
                    # calcular denominador
                    loc = ts[n-1]*(1+mu*new_params['Deltat']) + k*mu_J
                    scale = np.sqrt(ts[n-1]**2*new_params['Deltat']*sigma**2+k*sigma_J**2)
                    density_norm = scipy.stats.norm.pdf(x=ts[n],loc=loc,scale=scale)
                    mass_poisson = scipy.stats.poisson.pmf(k=k, mu=lbda)
                    den += density_norm*mass_poisson
                
                d_mu, d_sigma, d_lbda, d_mu_J, d_sigma_J = 0,0,0,0,0
                for k in range(k_MAX+1):
                    # calcular derivadas de f
                    M = ts[n-1]*(1+mu*new_params['Deltat']) + k*mu_J
                    Sigma = np.sqrt((ts[n-1]*sigma)**2*new_params['Deltat'] + (k*sigma_J)**2)
                    
                    d_mu += -np.sqrt(2)*new_params['Deltat']*ts[n-1]*(new_params['Deltat']*lbda)**k*(2*M - 2*ts[n])*np.exp(-new_params['Deltat']*lbda)*np.exp(-(-M + ts[n])**2/(2*Sigma**2))/(4*np.sqrt(np.pi)*Sigma**3*scipy.special.factorial(k))


                    d_sigma += new_params['Deltat']*ts[n-1]**2*sigma*(-np.sqrt(2)*(new_params['Deltat']*lbda)**k*np.exp(-new_params['Deltat']*lbda)*np.exp(-(-M + ts[n])**2/(2*Sigma**2))/(2*np.sqrt(np.pi)*Sigma**2*scipy.special.factorial(k)) + np.sqrt(2)*(new_params['Deltat']*lbda)**k*(-M + ts[n])**2*np.exp(-new_params['Deltat']*lbda)*np.exp(-(-M + ts[n])**2/(2*Sigma**2))/(2*np.sqrt(np.pi)*Sigma**4*scipy.special.factorial(k)))/np.sqrt(new_params['Deltat']*ts[n-1]**2*sigma**2 + k*sigma_J**2)

                    d_lbda += -np.sqrt(2)*new_params['Deltat']*(new_params['Deltat']*lbda)**k*np.exp(-new_params['Deltat']*lbda)*np.exp(-(-M + ts[n])**2/(2*Sigma**2))/(2*np.sqrt(np.pi)*Sigma*scipy.special.factorial(k)) + np.sqrt(2)*k*(new_params['Deltat']*lbda)**k*np.exp(-new_params['Deltat']*lbda)*np.exp(-(-M + ts[n])**2/(2*Sigma**2))/(2*np.sqrt(np.pi)*Sigma*lbda*scipy.special.factorial(k))

                    d_mu_J += -np.sqrt(2)*k*(new_params['Deltat']*lbda)**k*(2*M - 2*ts[n])*np.exp(-new_params['Deltat']*lbda)*np.exp(-(-M + ts[n])**2/(2*Sigma**2))/(4*np.sqrt(np.pi)*Sigma**3*scipy.special.factorial(k))


                    d_sigma_J += k*sigma_J*(-np.sqrt(2)*(new_params['Deltat']*lbda)**k*np.exp(-new_params['Deltat']*lbda)*np.exp(-(-M + ts[n])**2/(2*Sigma**2))/(2*np.sqrt(np.pi)*Sigma**2*scipy.special.factorial(k)) + np.sqrt(2)*(new_params['Deltat']*lbda)**k*(-M + ts[n])**2*np.exp(-new_params['Deltat']*lbda)*np.exp(-(-M + ts[n])**2/(2*Sigma**2))/(2*np.sqrt(np.pi)*Sigma**4*scipy.special.factorial(k)))/np.sqrt(new_params['Deltat']*ts[n-1]**2*sigma**2 + k*sigma_J**2)

                    
                jac += np.array([d_mu,d_sigma,d_lbda,d_mu_J,d_sigma_J])/den
            return -jac

        def EM():
            Theta_0 = np.ones(5)
            Theta_l = np.ones(5)
            
            def E(Theta):
                k_MAX = 1
                
                mu, sigma = Theta[0], Theta[1]
                lbda = Theta[2]
                mu_J, sigma_J = Theta[3], Theta[4]

                mu_l, sigma_l = Theta_l[0], Theta_l[1]
                lbda_l = Theta_l[2]
                mu_J_l, sigma_J_l = Theta_l[3], Theta_l[4]

                Q = 0
                for n in range(1, new_params['N']):
                    ksum = 0
                    for k in range(0, k_MAX+1):
                        
                        den = 0
                        # usamos parámetros de paso l para calcular q (parametros _l)
                        for kp in range(0, k_MAX+1):
                            loc = ts[n-1]*(1+mu_l*new_params['Deltat']) + kp*mu_J_l
                            scale = np.sqrt(ts[n-1]**2*new_params['Deltat']*sigma_l**2+kp*sigma_J_l**2)
                            density_norm = scipy.stats.norm.pdf(x=ts[n],loc=loc,scale=scale)
                            mass_poisson = scipy.stats.poisson.pmf(k=kp, mu=lbda)
                            den += density_norm*mass_poisson
                            
                        loc = ts[n-1]*(1+mu_l*new_params['Deltat']) + k*mu_J_l
                        scale = np.sqrt(ts[n-1]**2*new_params['Deltat']*sigma_l**2+k*sigma_J_l**2)
                        density_norm = scipy.stats.norm.pdf(x=ts[n],loc=loc,scale=scale)
                        mass_poisson = scipy.stats.poisson.pmf(k=k, mu=lbda)
                        q = density_norm*mass_poisson/den

                        # calculamos el valor de log(...)
                        loc = ts[n-1]*(1+mu*new_params['Deltat']) + k*mu_J
                        scale = np.sqrt(ts[n-1]**2*new_params['Deltat']*sigma**2+k*sigma_J**2)
                        density_norm = scipy.stats.norm.pdf(x=ts[n],loc=loc,scale=scale)
                        mass_poisson = scipy.stats.poisson.pmf(k=k, mu=lbda)
                        argument_log = density_norm*mass_poisson

                        ksum += q*np.log(argument_log)
    
                    Q += ksum
                return Q

            def M(Theta_0):
                def callback(Theta):
                    print(f"mu:\t\t{Theta[0]}")
                    print(f"sigma:\t\t{Theta[1]}")
                    print(f"lambda:\t\t{Theta[2]}")
                    print(f"mu_J:\t\t{Theta[3]}")
                    print(f"sigma_J:\t{Theta[4]}")
                    print("")

                bounds = [(-np.inf, np.inf), (0, np.inf), (0, np.inf), (-np.inf, np.inf), (0, np.inf)]

                print("Primera optimización...:")
                opt_res = scipy.optimize.minimize(lambda x: -E(x), Theta_0, method='L-BFGS-B',bounds=bounds,
                                                  options={'disp':True},callback=callback)
                if not opt_res.success:
                    print("ERROR: EM: no se ha podido optimizar en el paso M.")
                    return
                Theta_l = Theta_0
                Theta_new = opt_res.x
                print(Theta_new)
                while np.sum((Theta_new-Theta_l)**2)>0.1:
                    Theta_l = Theta_new
                    opt_res = scipy.optimize.minimize(lambda x: -E(x), Theta_l, method='L-BFGS-B',bounds=bounds,
                                                  options={'disp':True},callback=callback)
                    Theta_new = opt_res.x
                return Theta_new

            return M(Theta_0);

        return EM()
        
        

        bounds = [(-np.inf, np.inf), (0, np.inf), (0, np.inf), (-np.inf, np.inf), (0, np.inf)]
        
        opt_res = scipy.optimize.minimize(neg_log_likelihood, 5*np.ones(5), method='L-BFGS-B',
                                          callback=callback, jac=jac,bounds=bounds,
                                          options={'disp': True}, )

        if opt_res.success:
            print("Ajuste de parámetros realizado correctamente.")
            new_params['mu'] = opt_res.x[0]
            new_params['sigma'] = opt_res.x[1]
            new_params['lambda'] = opt_res.x[2]
            new_params['mu_J'] = opt_res.x[3]
            new_params['sigma_J'] = opt_res.x[4]
            self.informar_parametros(new_params)
        else:
            print("ERROR: no se han ajustado los parámetros correctamente.")

            
                    
        

        










            
        