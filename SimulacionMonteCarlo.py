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
        nones = sum([var not in param_dict.keys() for var in ('Deltat', 'N', 'T')])
        if  nones >= 2:
            print("ERROR: Hay que informar como mínimo dos de los tres siguientes parámetros:")
            print("Deltat, T, N")
            return False
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
                return False
        for var in ['mu', 'sigma', 'S0', 'M']:
            if var not in param_dict.keys():
                print(f"ERROR: {var} debe estar informada.")
                return False
            elif var != 'mu' and param_dict[var] < 0:
                print(f"ERROR: {var} debe ser mayor o igual que cero.")
                return False
        if param_dict['M'] == 0:
            print("WARNING: el número de simulaciones es cero.")
            return False

        return True

    
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
        

    def simular(self):
        if not self.confirmar_parametros_validos(self.param_dict):
            print("ERROR: parámetros inválidos. Abortando simulación.")
            return

        S = np.full([self.N+1, self.M], self.S0, dtype=np.dtype(float))
        Z = np.random.standard_normal([self.N, self.M])
        for i in range(self.N):
            S[i+1,:] = S[i,:]*(1 + self.mu * self.Deltat + self.sigma * np.sqrt(self.Deltat)*Z[i, :])
        return S
        

    def mostrar_parametros(self):
        # formatear esto debidamente
        print(self.param_dict)


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
                               


















            
        