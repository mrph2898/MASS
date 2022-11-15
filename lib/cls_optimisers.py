from typing import Optional
import numpy as np
import lib.anderson_acceleration as AA
from .base_opt import BaseSaddleOpt
from .problems import BaseSaddle



class SimGD(BaseSaddleOpt):
    def __init__(self,
                 problem: BaseSaddle,
                 x0: np.ndarray, 
                 y0: np.ndarray,
                 eps: float,
                 stopping_criteria: Optional[str],
                 params: dict
                ):
        super().__init__(problem, x0, y0, eps, stopping_criteria, params)
        if params is None:
            self.params = self._get_apdg_params(problem)
        
 
    @staticmethod
    def _get_apdg_params(problem: BaseSaddle):
        L = max(problem.L_x, problem.L_y, problem.L_xy)
        return {"lr": problem.mu_y / (4*L**2)}

    
    def step(self):
        gx, gy = self.problem.grad(self.x, self.y)
        self.x = self.x - self.params['lr'] * gx
        self.y = self.y + self.params['lr'] * gy
    
    
class AltGD(BaseSaddleOpt):
    def __init__(self,
                 problem: BaseSaddle,
                 x0: np.ndarray, 
                 y0: np.ndarray,
                 eps: float,
                 stopping_criteria: Optional[str],
                 params: dict
                ):
        super().__init__(problem, x0, y0, eps, stopping_criteria, params)
        if params is None:
            self.params = self._get_apdg_params(problem)
        
 
    @staticmethod
    def _get_apdg_params(problem: BaseSaddle):
        L = max(problem.L_x, problem.L_y, problem.L_xy)
        return {"lr": 1 / (2*L)}
        
        
    def step(self):
        g_x, _ = self.problem.grad(self.x, self.y)
        self.x = self.x - self.params['lr'] * g_x
        _, g_y = self.problem.grad(self.x, self.y)
        self.y = self.y + self.params['lr'] * g_y
    

class Avg(BaseSaddleOpt):
    def __init__(self,
                 problem: BaseSaddle,
                 x0: np.ndarray, 
                 y0: np.ndarray,
                 eps: float,
                 stopping_criteria: Optional[str],
                 params: dict
                ):
        super().__init__(problem, x0, y0, eps, stopping_criteria, params)
        self.xavg, self.yavg = self.x.copy(), self.y.copy()
        if params is None:
            self.params = self._get_apdg_params(problem)
        
 
    @staticmethod
    def _get_apdg_params(problem: BaseSaddle):
        L = max(problem.L_x, problem.L_y, problem.L_xy)
        return {"lr": 1 / (2*L)}
        
        
    def step(self):
        self.x = self.x - self.params['lr'] / np.sqrt(self.iter_count + 1)*(self.y)
        self.y = self.y + self.params['lr'] / np.sqrt(self.iter_count + 1)*(self.x)        
        self.xavg = self.xavg*(self.iter_count + 1)/(self.iter_count + 2) + self.x/(self.iter_count + 2)
        self.yavg = self.yavg*(self.iter_count + 1)/(self.iter_count + 2) + self.y/(self.iter_count + 2)  
        

class EG(BaseSaddleOpt):
    def __init__(self,
                 problem: BaseSaddle,
                 x0: np.ndarray, 
                 y0: np.ndarray,
                 eps: float,
                 stopping_criteria: Optional[str],
                 params: dict
                ):
        super().__init__(problem, x0, y0, eps, stopping_criteria, params)
        if params is None:
            self.params = self._get_apdg_params(problem)
        
 
    @staticmethod
    def _get_apdg_params(problem: BaseSaddle):
        L = max(problem.L_x, problem.L_y, problem.L_xy)
        return {"lr": 1 / (2*L)}
        

    def step(self):
        gx, gy = self.problem.grad(self.x, self.y)
        x_ = self.x - self.params['lr'] * gx
        y_ = self.y + self.params['lr'] * gy
        gx, gy = self.problem.grad(x_, y_)
        self.x = self.x - self.params['lr'] * gx
        self.y = self.y + self.params['lr'] * gy
        
        
class OMD(BaseSaddleOpt):
    def __init__(self,
                 problem: BaseSaddle,
                 x0: np.ndarray, 
                 y0: np.ndarray,
                 eps: float,
                 stopping_criteria: Optional[str],
                 params: dict
                ):
        super().__init__(problem, x0, y0, eps, stopping_criteria, params)
        x_l, y_l = 0.5*x0, 0.5*y0
        self.g_xl, self.g_yl = self.problem.grad(x_l,y_l)
        if params is None:
            self.params = self._get_apdg_params(problem)
        
 
    @staticmethod
    def _get_apdg_params(problem: BaseSaddle):
        L = max(problem.L_x, problem.L_y, problem.L_xy)
        return {"lr": 1 / (2*L)}

    def step(self):
        g_x, g_y = self.problem.grad(self.x,self.y)
        self.x = self.x - 2 * self.params['lr'] * g_x + self.params['lr'] * self.g_xl
        self.y = self.y + 2 * self.params['lr'] * g_y - self.params['lr'] * self.g_yl
        self.g_xl, self.g_yl =  g_x, g_y
    

class AltGDAAM(BaseSaddleOpt):
    def __init__(self,
                 problem: BaseSaddle,
                 x0: np.ndarray, 
                 y0: np.ndarray,
                 eps: float,
                 stopping_criteria: Optional[str],
                 params: dict
                ):
        super().__init__(problem, x0, y0, eps, stopping_criteria, params)
        self.fp = np.vstack((self.x, self.y))
        self.aa = AA.numpyAA(2, params['k'], type2=params['type2'], reg=params['reg'])
        if params is None:
            self.params = self._get_apdg_params(problem)
        
 
    @staticmethod
    def _get_apdg_params(problem: BaseSaddle):
        L = max(problem.L_x, problem.L_y, problem.L_xy)
        return {"lr": 1 / (2*L)}
        

    def step(self):
        fpprev = np.copy(self.fp)
        g_x, _ = self.problem.grad(self.x, self.y)
        x_ = (1 - self.params['gamma']) * self.x - self.params['lr'] * g_x
        _, g_y = self.problem.grad(x_, self.y)
        y_ = self.y + self.params['lr'] * g_y
        self.fp = np.vstack((x_, y_))
        self.fp = self.aa.apply(fpprev, self.fp)
        self.x, self.y = self.fp[0],self.fp[1]
        
        
class APDG(BaseSaddleOpt):
    def __init__(self,
                 problem: BaseSaddle,
                 x0: np.ndarray, 
                 y0: np.ndarray,
                 eps: float,
                 stopping_criteria: Optional[str],
                 params: dict
                ):
        super().__init__(problem, x0, y0, eps, stopping_criteria, params)
        self.x_f, self.y_f = x0.copy(), y0.copy()
        self.y_prev = y0.copy() 
        self.A = problem.A
        if params is None:
            self.params = self._get_apdg_params(problem)
        
        
    @staticmethod
    def _get_apdg_params(problem: BaseSaddle):
        """
        Setting the optimal params from https://arxiv.org/abs/2112.15199v2 , pp.12-18
        """
        L_x, L_y, mu_x, mu_y = problem.L_x, problem.L_y, problem.mu_x, problem.mu_y
        L_xy, mu_xy, mu_yx = problem.L_xy, problem.mu_xy, problem.mu_yx
        rho_a, rho_b, rho_c, rho_d = -np.inf, -np.inf, -np.inf, -np.inf

        if (mu_x > 0) and (mu_y > 0):
            delta = (mu_y / mu_x)**.5
            sigma_x = (mu_x / (2 * L_x))**.5
            sigma_y = (mu_y / (2 * L_y))**.5
            rho_a = (4 + 4 * max((L_x / mu_x)**.5,
                            (L_y / mu_y)**.5,
                            L_xy / (mu_x * mu_y)**.5)) ** (-1)
        elif (mu_x > 0) and (mu_y == 0):
            delta = (mu_xy**2 / (2*mu_x*L_x))**.5
            sigma_x = (mu_x / (2 * L_x))**.5
            sigma_y = min(1, (mu_xy**2 / (4*L_x*L_y))**.5)
            rho_b = (4 + 8 * max((L_x * L_y)**.5 / mu_xy,
                            L_xy / mu_xy * (L_x / mu_x)**.5,
                            L_xy**2 / mu_xy**2)) ** (-1)
        elif (mu_x == 0) and (mu_y > 0):
            delta = (2*mu_y**2 * L_y / mu_yx**2)**.5
            sigma_x = min(1, (mu_yx**2 / (4*L_x*L_y))**.5)
            sigma_y = (mu_y / (2 * L_y))**.5
            rho_c = (4 + 8 * max((L_x * L_y)**.5 / mu_yx,
                                 L_xy / mu_yx *(L_y / mu_y)**.5,
                                 L_xy**2 / mu_xy**2)) ** (-1)
        elif (mu_x == 0) and (mu_y == 0):
            delta = (mu_xy / mu_yx)*(L_y / L_x)**.5
            sigma_x = min(1, (mu_yx**2 / (4*L_x*L_y))**.5)
            sigma_y = min(1, (mu_yx**2 / (4*L_x*L_y))**.5)
            rho_d = (2 + 8 * max((L_x * L_y)**.5 * L_xy / (mu_xy * mu_yx),
                            L_xy**2 / mu_yx**2,
                            L_xy**2 / mu_xy**2)) ** (-1)

        theta = 1 - max(rho_a, rho_b, rho_c, rho_d)
        eta_x = min(1 / (4 * (mu_x + L_x * sigma_x)), delta/(4*L_xy))
        eta_y = min(1 / (4 * (mu_y + L_y * sigma_y)), 1/(4*L_xy * delta))
        tau_x = (sigma_x**(-1) + 0.5)**(-1)
        tau_y = (sigma_y**(-1) + 0.5)**(-1)
        alpha_x = mu_x
        alpha_y = mu_y
        beta_x = min(1/(2*L_y), 1/(1*eta_x*L_xy**2))
        beta_y = min(1/(2*L_x), 1/(1*eta_y*L_xy**2))
        opt_params = {"sigma_x": sigma_x,
                      "sigma_y": sigma_y, 
                      "theta": theta,
                      "tau_x": tau_x,
                      "tau_y": tau_y,
                      "eta_x": eta_x,
                      "eta_y": eta_y,
                      "alpha_x": alpha_x,
                      "alpha_y": alpha_y,
                      "beta_x": beta_x,
                      "beta_y": beta_y,
                     }
        return opt_params
        
        
    def step(self):
        y_m = self.y + self.params['theta'] * (self.y - self.y_prev)
        x_g = self.params['tau_x'] * self.x + (1 - self.params['tau_x']) * self.x_f
        y_g = self.params['tau_y'] * self.y + (1 - self.params['tau_y']) * self.y_f

        x_g = self.proj_x(x_g)
        y_g = self.proj_y(y_g)

        grad_x, grad_y = self.problem.fg_grads(x_g, y_g)
        self.x = (self.x + self.params['eta_x'] * self.params['alpha_x'] * (x_g - self.x) -
             self.params['eta_x'] * self.params['beta_x'] * (self.A.T.dot(self.A.dot(self.x) - grad_y)) - 
             self.params['eta_x'] * (grad_x + self.A.T.dot(y_m))
            )
        self.y = (self.y + self.params['eta_y'] * self.params['alpha_y'] * (y_g - self.y) - 
             self.params['eta_y'] * self.params['beta_y'] * (self.A.dot(self.A.T.dot(self.y) + grad_x)) -
             self.params['eta_y'] * (grad_y - self.A.dot(self.x))
            )

        self.x_f = x_g + self.params['sigma_x'] * (self.x - self.x_hist[-1])
        self.y_f = y_g + self.params['sigma_y'] * (self.y - self.y_hist[-1])

        self.x_f = self.proj_x(self.x_f)
        self.y_f = self.proj_y(self.y_f)
        
        self.y_prev = self.y_hist[-1]
        
        
class LPD(BaseSaddleOpt):
    def __init__(self,
                 problem: BaseSaddle,
                 x0: np.ndarray, 
                 y0: np.ndarray,
                 eps: float,
                 stopping_criteria: Optional[str],
                 params: dict
                ):
        super().__init__(problem, x0, y0, eps, stopping_criteria, params)
        self.x_prev, self.y_prev = self.x, self.y
        self._grad_f = lambda x: problem.grad_f(x) - problem.mu_x*x
        self._grad_h = lambda y: problem.grad_g(y) - problem.mu_y*y

        self.xp = self._grad_f(x0)
        self.yp = self._grad_h(y0)
        self.xp_prev, self.yp_prev = self.xp, self.yp
        self.bx, self.by = self.x, self.y

        self.grad_bx, self.grad_by = self._grad_f(self.bx), self._grad_h(self.by)
        self.grad_bx_prev, self.grad_by_prev = self.grad_bx, self.grad_by
        
        if params is None:
            self.params = self._get_lpd_params(problem)
        
        
    @staticmethod
    def _get_lpd_params(problem: BaseSaddle):
        theta_x, theta_y, theta_xp, theta_yp, = None, None, None, None
        stepsize_x, stepsize_y, stepsize_xp, stepsize_yp, _stepsize_x, _stepsize_y = None, None, None, None, None, None
        kappa_x = (-1+problem.L_x/problem.mu_x)
        kappa_y = (-1+problem.L_y/problem.mu_y)
        if problem.mu_x != 0.0 and problem.mu_y != 0.0:
            gamma = 1 + (
                kappa_x**0.5 + 
                2*problem.L_xy / ((problem.mu_x*problem.mu_y)**0.5) +
                kappa_y**0.5
              )**(-1.0)

            stepsize_x = (1/problem.mu_x)*(
                kappa_x**0.5 + 2*problem.L_xy/((problem.mu_x*problem.mu_y)**0.5)
            )**(-1.0)

            stepsize_y = (1/problem.mu_y)*(
                kappa_y**0.5 + 2*problem.L_xy/((problem.mu_x*problem.mu_y)**0.5)
            )**(-1.0)

            if kappa_x == 0.0:
                stepsize_xp = 0.0
            else:
                stepsize_xp = (kappa_x**0.5)**(-1.0)

            if kappa_y == 0.0:
                stepsize_yp = 0.0
            else:
                stepsize_yp = (kappa_y**0.5)**(-1.0)

            theta_x = theta_y = theta_xp = theta_yp = 1/gamma
        else:
            _stepsize_x = 1/2/(problem.L_x - problem.mu_x + 1.e-32)
            _stepsize_y = 1/2/(problem.L_y - problem.mu_y + 1.e-32)

            if problem.mu_x == 0.0 and problem.mu_y != 0.0:
                _stepsize_x = (1./_stepsize_x + 16*problem.L_xy/problem.mu_y)**-1.0
            if problem.mu_x != 0.0 and problem.mu_y == 0.0:
                _stepsize_y = (1./_stepsize_y + 16*problem.L_xy/problem.mu_x)**-1.0
                
        opt_params = {"theta_x": theta_x,
                      "theta_y": theta_y, 
                      "theta_xp": theta_xp,
                      "theta_yp": theta_yp,
                      "stepsize_x": stepsize_x,
                      "stepsize_y": stepsize_y,
                      "stepsize_xp": stepsize_xp,
                      "stepsize_yp": stepsize_yp,
                      "_stepsize_x": _stepsize_x,
                      "_stepsize_y": _stepsize_y,
                     }
        return opt_params
    
    
    def step(self):
        k = self.iter_count

        if self.problem.mu_x == 0.0 or self.problem.mu_y == 0.0:
            self.params['theta_x'] = self.params['theta_y'] = self.params['theta_xp'] = self.params['theta_yp'] = k/(k+1)
            self.params['stepsize_x'] = (1.0/(k+1)/self.params['_stepsize_x'] + k*self.problem.mu_x/2.0)**-1.0
            self.params['stepsize_y'] = (1.0/(k+1)/self.params['_stepsize_y'] + k*self.problem.mu_y/2.0)**-1.0

        tx_next = self.x + self.params['theta_x']*(self.x - self.x_prev)
        ty_next = self.y + self.params['theta_y']*(self.y - self.y_prev)

        tgrad_x_next = self.grad_bx + self.params['theta_xp']*(self.grad_bx - self.grad_bx_prev)
        tgrad_y_next = self.grad_by + self.params['theta_yp']*(self.grad_by - self.grad_by_prev)

        x_next = (self.x - self.params['stepsize_x']*(self.problem.A.T.dot(ty_next) + tgrad_x_next))/(1.+self.params['stepsize_x']*self.problem.mu_x)
        y_next = (self.y + self.params['stepsize_y']*(self.problem.A.dot(tx_next) - tgrad_y_next))/(1.+self.params['stepsize_y']*self.problem.mu_y)

        if k != 0 and (self.problem.mu_x == 0.0 or self.problem.mu_y == 0.0):
            self.params['stepsize_xp'] = 2.0/k
            self.params['stepsize_yp'] = 2.0/k

        if k == 0 and (self.problem.mu_x == 0.0 or self.problem.mu_y == 0.0):
            bx_next =  x_next
            by_next =  y_next
        else:
            bx_next =  (self.bx+ self.params['stepsize_xp']*x_next)/(1+self.params['stepsize_xp'])
            by_next =  (self.by+ self.params['stepsize_yp']*y_next)/(1+self.params['stepsize_yp'])

        bx_next = self.proj_x(bx_next)
        by_next = self.proj_y(by_next)

        grad_bx_next = self._grad_f(bx_next)
        grad_by_next = self._grad_h(by_next)

    # update iterates
        self.x_prev, self.x = self.x, x_next
        self.y_prev, self.y = self.y, y_next

        self.bx = bx_next
        self.by = by_next

        self.grad_bx_prev, self.grad_bx = self.grad_bx, grad_bx_next
        self.grad_by_prev, self.grad_by = self.grad_by, grad_by_next
        

