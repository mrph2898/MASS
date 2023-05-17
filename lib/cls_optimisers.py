from typing import Optional, Union, Callable
import numpy as np
import numpy.linalg as LA
from autograd import grad
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
        
        self._step_complexity = 2+2+problem.gradx_complexity+problem.grady_complexity
        if params is None:
            self.params = self._get_simgd_params(problem)
        
 
    @staticmethod
    def _get_simgd_params(problem: BaseSaddle):
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
        
        self._step_complexity = 2+2+problem.gradx_complexity+problem.grady_complexity
        if params is None:
            self.params = self._get_altgd_params(problem)
        
 
    @staticmethod
    def _get_altgd_params(problem: BaseSaddle):
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
        
        self._step_complexity = 5+5+7+7
        if params is None:
            self.params = self._get_avg_params(problem)
        
 
    @staticmethod
    def _get_avg_params(problem: BaseSaddle):
        L = max(problem.L_x, problem.L_y, problem.L_xy)
        return {"lr": 1 / (2*L)}
        
        
    def step(self):
        k = self.iter_count
        self.x = self.x - self.params['lr'] / np.sqrt(k + 1)*(self.y)
        self.y = self.y + self.params['lr'] / np.sqrt(k + 1)*(self.x)        
        self.xavg = self.xavg*(k + 1)/(k + 2) + self.x/(k + 2)
        self.yavg = self.yavg*(k + 1)/(k + 2) + self.y/(k + 2)  
        

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
        
        self._step_complexity = 2+2+2+2+(problem.gradx_complexity+problem.grady_complexity)*2
        if params is None:
            self.params = self._get_eg_params(problem)
        
 
    @staticmethod
    def _get_eg_params(problem: BaseSaddle):
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
        
        self._step_complexity = 5+5+problem.gradx_complexity+problem.grady_complexity
        if params is None:
            self.params = self._get_omd_params(problem)
        
 
    @staticmethod
    def _get_omd_params(problem: BaseSaddle):
        L = max(problem.L_x, problem.L_y, problem.L_xy)
        return {"lr": problem.mu_y / (4*L**2)}

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
        
        # TODO: Count AA complexity
        self._step_complexity = None
        if params is None:
            self.params = self._get_altgdaam_params(problem)
        self.aa = AA.numpyAA(2, self.params['k'], 
                             type2=self.params['type2'],
                             reg=self.params['reg'])
        
 
    @staticmethod
    def _get_altgdaam_params(problem: BaseSaddle):
        L = max(problem.L_x, problem.L_y, problem.L_xy)
        return {"lr": 1 / (2*L), 
                 "k": 10,
                 "type2": True,
                 "reg": 1e-10,
                 "gamma": 1e-26
                }
        

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
        
        N, M = self.A.shape
        self._step_complexity = (3+4+4+(7+(N+1)*M+2+1+M)+(7+(M+1)*N+3+N)+
                                 3+3+problem.grad_f_complexity+problem.grad_g_complexity)
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
        
        N, M = problem.A.shape
        self._step_complexity = (3+3+3+3+(2+M+1+3)+(2+N+1+3)+3+3+
                                 problem.grad_f_complexity+2+problem.grad_g_complexity+2)
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
        

class SepMiniMax(BaseSaddleOpt):
    def __init__(self,
                 problem: BaseSaddle,
                 x0: np.ndarray, 
                 y0: np.ndarray,
                 eps: float,
                 stopping_criteria: Optional[str],
                 params: dict
                ):
        """
        Implementation of Sidford et al. Separable minimax optimisation.
        https://arxiv.org/pdf/2202.04640.pdf
        """
        super().__init__(problem, x0, y0, eps, stopping_criteria, params)
        
        N,M = problem.A.shape
        self._step_complexity = (M+N+3+3+4+4+6+6+M+N+3+3+12+12+7+7+
                                 (problem.grad_f_complexity+problem.grad_g_complexity)*2
                                )
        if params is None:
            self.params = self._get_smm_params(problem)
        self.x_f = x0.copy()
        self.y_g = y0.copy()
        
 
    @staticmethod
    def _get_smm_params(problem: BaseSaddle):
        """
        Optimal lambda is taken from Lemma 6 (Relative Lipschitzness)
        """
        _lambda = (1 + (problem.L_x/problem.mu_x)**.5 +(problem.L_y/problem.mu_y)**.5 +
                   problem.L_xy/(problem.mu_x*problem.mu_y)**.5
                  ) 
        return {"lambda": _lambda}
        
        
    def step(self):
        _df, _dg = self.problem.fg_grads(self.x_f, self.y_g)
        bilinear_dx, bilinear_dy = self.problem.A.T.dot(self.y), self.problem.A.dot(self.x)
        Phi_x = self.problem.mu_x*self.x + _df + bilinear_dx
        Phi_y = self.problem.mu_y*self.y + _dg - bilinear_dy
        
        # Gradient step
        x_half = self.x - (self.params["lambda"]*self.problem.mu_x)**-1 * Phi_x
        y_half = self.y - (self.params["lambda"]*self.problem.mu_y)**-1 * Phi_y
        xf_half = (1 - 1/self.params["lambda"])*self.x_f + 1/self.params["lambda"]*self.x
        yg_half = (1 - 1/self.params["lambda"])*self.y_g + 1/self.params["lambda"]*self.y
        
        _df, _dg = self.problem.fg_grads(xf_half, yg_half)
        bilinear_dx, bilinear_dy = self.problem.A.T.dot(y_half), self.problem.A.dot(x_half)
        Phi_x = self.problem.mu_x*x_half + _df + bilinear_dx
        Phi_y = self.problem.mu_y*y_half + _dg - bilinear_dy
        
        # Extragradient step
        self.x = (1/(1 + self.params["lambda"]) * x_half +
                  self.params["lambda"]/(1 + self.params["lambda"]) * self.x - 
                  1/((1 + self.params["lambda"])*self.problem.mu_x)*Phi_x
                 )
        self.y = (1/(1 + self.params["lambda"]) * y_half +
                  self.params["lambda"]/(1 + self.params["lambda"]) * self.y - 
                  1/((1 + self.params["lambda"])*self.problem.mu_y)*Phi_y
                 )
        
        self.x_f = (self.params["lambda"]/(1 + self.params["lambda"]) * self.x_f +
                    1/(1 + self.params["lambda"]) * x_half
                   )
        self.y_g = ((self.params["lambda"]/(1 + self.params["lambda"]) * self.y_g +
                     1/(1 + self.params["lambda"]) * y_half))
        
        
class FOAM(BaseSaddleOpt):
    """
    FOAM algorithm from https://arxiv.org/pdf/2205.05653.pdf 
    """
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
        self.z_f = x0.copy()
        self.z = x0.copy()
        self.A = problem.A
        
        N, M = self.A.shape
        self._while_step_complexity = (3*N+3*M+1+(3+N)+(3+M)+(7+5)+(7+5)+(6+5)+(6+5)+(9+5+1)+(9+5+1)+
                                      problem.prox_f_complexity+problem.prox_g_complexity)
        self._zero_step_complexity = (4+4+3+(3+5+1)+(3+5+1)+(6+5+1)+(6+5+1)+3+4+10+9+3+
                                      problem.prox_f_complexity+problem.prox_g_complexity+
                                      problem.grad_f_complexity+problem.grad_g_complexity
                                     )
        self._step_complexity = self._zero_step_complexity
        if params is None:
            self.params = self._get_foam_params(problem)
        
        
    @staticmethod
    def _get_foam_params(problem: BaseSaddle):
        """
        Setting the optimal params from pp.8-10
        """
        L_x, L_y, mu_x, mu_y = problem.L_x, problem.L_y, problem.mu_x, problem.mu_y
        L_xy, mu_xy, mu_yx = problem.L_xy, problem.mu_xy, problem.mu_yx
        L = max(L_x, L_y, L_xy)
    
        theta = 8/mu_x
        eta_z = 0.5/mu_x
        gamma_x = 8/mu_x
        gamma_y = theta
        M = 2*max(gamma_x*L, gamma_y*(L + theta**-1))
        alpha = min(1, (theta*mu_y)**.5)
        
        eta_y = min(1/(2*mu_y), theta/(2*alpha))
        _lambda = 1/(5**.5 * M)

        opt_params = {
                      "theta": theta,
                      "gamma_x": gamma_x,
                      "gamma_y": gamma_y,
                      "eta_z": eta_z,
                      "eta_y": eta_y,
                      "alpha": alpha,
                      "lambda": _lambda
                     }
        return opt_params
        
    def a_x(self, x, y, z_g):
        g_x, _ = self.problem.grad(x, y)
        return g_x - 0.5*self.problem.mu_x*x - 0.5*z_g
    
    def a_y(self, x, y, y_g):
        _, g_y = self.problem.grad(x, y)
        return -g_y - 1/self.params["theta"] * (y - y_g)
    
    def beta(self, t):
        return 2/(t + 3)
        
    def step(self):
        self._step_complexity = self._zero_step_complexity
        _alpha = self.params['alpha']
        _theta = self.params["theta"] 
        _eta_z = self.params["eta_z"] 
        _eta_y = self.params["eta_y"] 
        _gamma_x = self.params["gamma_x"] 
        _gamma_y = self.params["gamma_y"]
        _lambda = self.params["lambda"]
        
        z_g = _alpha * self.z + (1 - _alpha) * self.z_f
        y_g = _alpha * self.y + (1 - _alpha) * self.y_f
        
        _x_m1, _y_m1 = -self.problem.mu_x**-1 * z_g, y_g
        
        x_k_0 = self.problem.prox_f(_x_m1 - _gamma_x*_lambda*self.a_x(_x_m1, _y_m1, z_g),
                                    scale=_gamma_x*_lambda)
        y_k_0 = self.problem.prox_g(_y_m1 - _gamma_y*_lambda*self.a_y(_x_m1, _y_m1, y_g),
                                    scale=_gamma_y*_lambda)
        x_k = x_k_0.copy()
        y_k = y_k_0.copy()
        b_x_k = (_gamma_x*_lambda)**-1 * (_x_m1 - _gamma_x*_lambda*self.a_x(_x_m1, _y_m1, z_g) - x_k_0)
        b_y_k = (_gamma_y*_lambda)**-1 * (_y_m1 - _gamma_y*_lambda*self.a_y(_x_m1, _y_m1, y_g) - y_k_0)
        
        t = 0
        while (_gamma_x*LA.norm(self.a_x(x_k, y_k, z_g) + b_x_k)**2 + 
               _gamma_y*LA.norm(self.a_y(x_k, y_k, y_g) + b_y_k)**2) > (
            1/_gamma_x*LA.norm(x_k - _x_m1)**2 + 
            1/_gamma_y*LA.norm(y_k - _y_m1)**2
        
        ):
            self._step_complexity += self._while_step_complexity
            x_k_half = x_k + self.beta(t)*(x_k_0 - x_k) - _gamma_x*_lambda*(self.a_x(x_k, y_k, z_g) + b_x_k)  
            y_k_half = y_k + self.beta(t)*(y_k_0 - y_k) - _gamma_y*_lambda*(self.a_y(x_k, y_k, y_g) + b_y_k)  
            x_k_next = self.problem.prox_f(x_k + self.beta(t)*(x_k_0 - x_k) -
                                           _gamma_x*_lambda*self.a_x(x_k_half, y_k_half, z_g),
                                           scale=_gamma_x*_lambda
                                          )
            y_k_next = self.problem.prox_g(y_k + self.beta(t)*(y_k_0 - y_k) -
                                           _gamma_y*_lambda*self.a_y(x_k_half, y_k_half, y_g),
                                           scale=_gamma_y*_lambda
                                          )
            b_x_k = (_gamma_x*_lambda)**-1 * (x_k + self.beta(t)*(x_k_0 - x_k) -
                                              _gamma_x*_lambda*self.a_x(x_k_half, y_k_half, z_g) - x_k_next)
            b_y_k = (_gamma_y*_lambda)**-1 * (y_k + self.beta(t)*(y_k_0 - y_k) -
                                              _gamma_y*_lambda*self.a_y(x_k_half, y_k_half, y_g) - y_k_next)
            x_k, y_k = x_k_next, y_k_next
            t += 1
        t_k = t    
        
        self.x_f = x_k
        self.y_f = y_k
        
        g_x, g_y = self.problem.grad(self.x_f, self.y_f)
        self.z_f = g_x - self.problem.mu_x*self.x_f + b_x_k
        
        w_f = -g_y - self.problem.mu_y*self.y_f + b_y_k
        self.z = (self.z + _eta_z*self.problem.mu_x**-1 * (self.z_f - self.z) - 
                  _eta_z*(self.x_f + self.problem.mu_x**-1 * self.z_f))
        self.y = (self.y + _eta_y*self.problem.mu_y**-1 * (self.y_f - self.y) - 
                  _eta_y*(w_f + self.problem.mu_y * self.y_f))

        # x update only for saving the current x for metrics
        self.x = -self.problem.mu_x**-1 * self.z   
        self.x = self.proj_x(self.x)
        
        self.y = self.proj_y(self.y)
        

class AccEG(BaseSaddleOpt):
    def __init__(self,
                 problem: BaseSaddle,
                 inner_optimiser: BaseSaddleOpt,
                 inner_max_iter: int,
                 x0: np.ndarray, 
                 y0: np.ndarray,
                 eps: float,
                 stopping_criteria: Optional[str],
                 params: dict
                ):
        super().__init__(problem, x0, y0, eps, stopping_criteria, params)
        
        if params is None:
            self.params = self._get_acceg_params(problem)
        self.x_f = x0.copy()
        self.inner_opt = inner_optimiser
        self.inner_max_iter = inner_max_iter
        
 
    @staticmethod
    def _get_acceg_params(problem: BaseSaddle):
        _mu = min(problem.mu_x, problem.mu_y)
        L_p = problem.L_x
        tau = min(1, _mu**.5/(2*L_p**.5))
        theta = 0.5/L_p
        eta = min(0.5/_mu, 0.5/(_mu*L_p)**.5)
        alpha = _mu
        return {"tau": tau,
                "theta": theta,
                "eta": eta,
                "alpha": alpha
               }
    
    
    def step(self):
        x_g = self.params["tau"]*self.x + (1 - self.params["tau"])*self.x_f
        
        subproblem = BaseSaddle(A=self.problem.A)
        # min max f(x) + <y, Ax> - g(y)
        
        grad_p, _ = self.problem.fg_grads(x_g, self.y)
        subproblem.f = lambda x: 0.5/self.params["theta"]*x@x + (grad_p - 1/self.params["theta"]*x_g)@x
        subproblem.mu_x = subproblem.L_x = self.params["theta"]**-1
        subproblem._optimiser_params = self.params
        subproblem._optimiser_params["right_part"] = 1/self.params["theta"]*x_g + grad_p
        subproblem.g = self.problem.g
        subproblem.C = self.problem.C 
        subproblem.mu_y = self.problem.mu_y
        subproblem.L_y = self.problem.L_y
        subproblem.mu_xy = self.problem.mu_xy 
        subproblem.mu_yx = self.problem.mu_yx 
        subproblem.L_xy = self.problem.L_xy
        subproblem.xopt, subproblem.yopt = self.problem.xopt,self.problem.yopt
        subproblem.primal_func = self.problem.primal_func
        subproblem.dual_func = lambda y, x: subproblem.F(-self.params["theta"]*(grad_p - 1/self.params["theta"]*x_g + subproblem.A.T@y), y)
        subproblem.prox_f = lambda v, scale: (v + 1/self.params["theta"]*x_g - grad_p)*self.params["theta"]/(1 + self.params["theta"])
        subproblem.prox_g = self.problem.prox_g
        subproblem.grad_f = grad(subproblem.f)
        subproblem.grad_g = grad(subproblem.g)
        
        inner_optimiser = self.inner_opt(problem=subproblem, x0=x_g,
                                         y0=self.y, eps=self.eps,
                                         stopping_criteria='sliding',
                                         params=None
                                        )
        
        _loss, _, _ = inner_optimiser(max_iter=self.inner_max_iter,
                                         verbose=0)

        self.x_f = inner_optimiser.x
        self.y = inner_optimiser.y
        
        # TODO: take grad without y
        gx, _ = self.problem.fg_grads(self.x_f, inner_optimiser.y)
        _A = self.problem.A
        _C = self.problem.C
        gx += 3/4*_A.T @ LA.inv(_C) @ _A @ self.x_f
        self.x = (self.x + 
                  self.params['eta']*self.params['alpha']*(self.x_f - self.x) - 
                  self.params['eta']*gx
                 ) 
        
        
class AcceleratedEG(BaseSaddleOpt):
    def __init__(self,
                 problem: BaseSaddle,
                 inner_optimiser: BaseSaddleOpt,
                 inner_max_iter: int,
                 x0: np.ndarray, 
                 y0: np.ndarray,
                 eps: float,
                 stopping_criteria: Optional[str],
                 params: dict
                ):
        super().__init__(problem, x0, y0, eps, stopping_criteria, params)
        
        N,M = problem.A.shape
        self._zero_step_complexity = (4+4+8+8+2+(3+N)+(3+M)+5+5+(2+M)+(2+N)+3+3+
                                      problem.grad_f_complexity+problem.grad_g_complexity)
        self._step_complexity = self._zero_step_complexity
        if params is None:
            self.params = self._get_acceg_params(problem)
        self.x_f = x0.copy()
        self.y_f = y0.copy()
        self.inner_opt = inner_optimiser
        self.inner_max_iter = inner_max_iter
        
 
    @staticmethod
    def _get_acceg_params(problem: BaseSaddle):
        _mu = min(problem.mu_x, problem.mu_y)
        # L_p = max(problem.L_x, problem.L_y)
        if problem.L_x/problem.mu_x > problem.L_y/problem.mu_y:
            alpha = min(1, (problem.mu_x / problem.L_x)**.5)
            eta_x = min((3*problem.mu_x)**-1, (3*alpha*problem.L_x)**-1)
            eta_y = problem.mu_x/problem.mu_y * eta_x
        else:
            alpha = min(1, (problem.mu_y / problem.L_y)**.5)
            eta_y = min((3*problem.mu_y)**-1, (3*alpha*problem.L_y)**-1)
            eta_x = problem.mu_y/problem.mu_x * eta_y

        return {"eta_x": eta_x,
                "eta_y": eta_y,
                "alpha": alpha
               }
    
    
    def step(self):
        self._step_complexity = self._zero_step_complexity
        x_g = self.params["alpha"]*self.x + (1 - self.params["alpha"])*self.x_f
        y_g = self.params["alpha"]*self.y + (1 - self.params["alpha"])*self.y_f
        
        subproblem = BaseSaddle(A=self.problem.A)
        # min max f(x) + <y, Ax> - g(y)
        grad_f, grad_g = self.problem.fg_grads(x_g, y_g)
        subproblem.f = lambda x: self.problem.f(x_g) + grad_f@(x - x_g) + 1/2/self.params["eta_x"] * (x - self.x).T@(x - self.x) 
        subproblem.g = lambda y: self.problem.g(y_g) + grad_g@(y - y_g) + 1/2/self.params["eta_y"] * (y - self.y).T@(y - self.y) 
        subproblem.mu_x = subproblem.L_x = (self.params["eta_x"])**-1
        subproblem.mu_y = subproblem.L_y = (self.params["eta_y"])**-1
        subproblem._optimiser_params = self.params
        subproblem.mu_xy = self.problem.mu_xy 
        subproblem.mu_yx = self.problem.mu_yx 
        subproblem.L_xy = self.problem.L_xy
    
        u_opt = LA.solve(np.block([[np.eye(self.x.shape[0]), self.params["eta_x"]*subproblem.A.T],
                                  [-self.params["eta_y"]*subproblem.A, np.eye(self.y.shape[0])]]), 
                         np.hstack([self.x - self.params["eta_x"]*grad_f,
                                    self.y - self.params["eta_y"]*grad_g
                                   ])
                        )
        subproblem.xopt, subproblem.yopt = u_opt[: self.x.shape[0]], u_opt[self.x.shape[0]: ]
        subproblem.primal_func = lambda x, y: subproblem.F(self.x, self.y - self.params["eta_y"]*(grad_g - subproblem.A@x))
        subproblem.dual_func = lambda y, x: subproblem.F(self.x - self.params["eta_x"]*(grad_f + subproblem.A.T@y), y)
        subproblem.prox_f = lambda v, scale: (self.x - self.params["eta_x"]*(grad_f - v))/(1 + self.params["eta_x"])
        subproblem.prox_g = lambda v, scale: (self.y - self.params["eta_y"]*(grad_g - v))/(1 + self.params["eta_y"])
        subproblem.grad_f = grad(subproblem.f)
        subproblem.grad_g = grad(subproblem.g)
        N, M = self.problem.A.shape
        subproblem.gradx_complexity = 1+1+2+M
        subproblem.grady_complexity = 1+1+2+N
        subproblem.grad_f_complexity = 1+1+2
        subproblem.grad_g_complexity = 1+1+2
        subproblem.prox_f_complexity = 1+1+1+1+1
        subproblem.prox_g_complexity = 1+1+1+1+1
        
        inner_optimiser = self.inner_opt(problem=subproblem, x0=self.x,
                                         y0=self.y, eps=self.eps,
                                         stopping_criteria='accel_sliding',
                                         params=None
                                        )
        
        _loss, _, _ = inner_optimiser(max_iter=self.inner_max_iter,
                                      verbose=0)

        self._step_complexity += sum(inner_optimiser.all_metrics["step_complexity"])
        _x_next = self.x - self.params['eta_x']*(grad_f + self.problem.A.T@inner_optimiser.y)
        _y_next = self.y - self.params['eta_y']*(grad_g - self.problem.A@inner_optimiser.x)
        
        self.x_f = x_g + self.params["alpha"]*(inner_optimiser.x - self.x)
        self.y_f = y_g + self.params["alpha"]*(inner_optimiser.y - self.y)
        
        self.x = _x_next
        self.y = _y_next