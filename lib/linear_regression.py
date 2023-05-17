from typing import List, Optional
from autograd import grad, jacobian, elementwise_grad
import autograd.numpy as np
from numpy import linalg as LA
from scipy.linalg import pinv
import scipy.linalg as sla
from scipy.stats import ortho_group
import scipy


class LinearRegression:
  """
  Minimize x \\in \\reals^dx
  and
  Maximize y \\in \\reals^dy:
  :
  F(x, y) = f(x) + <y, Ax> - g(y)
  """
  def __init__(
    self, l, A, y, xopt=None, yopt=None
    ):

    # L_x =  - Lipschitz constant of grad_x w.r.t x
    # mu_x = - Strong convexity constant of grad_x w.r.t x
    # L_xy =  - Lipschitz constant of grad_x w.r.t y
    # mu_xy = - "Strong convexity" constant of grad_x w.r.t y
    # L_yx =  - Lipschitz constant of grad_y w.r.t x
    # mu_yx = - "Strong concavity" constant of grad_y w.r.t x
    # L_y = - Lipschitz constant of grad_y w.r.t y
    # mu_y = - Strong concavity constant of grad_y w.r.t y

    self.rho = l
    self.L_x = l
    self.mu_x = self.L_x

    self.n = A.shape[0]
    self._y = y

    self.A = A

    eigvalsxy = LA.eigvalsh(self.A @ self.A.T)
    self.L_xy, self.mu_xy = np.sqrt(eigvalsxy[-1]), np.sqrt(np.abs(eigvalsxy).min())

    eigvalsyx = LA.eigvalsh(self.A.T @ self.A)
    self.L_yx, self.mu_yx = np.sqrt(eigvalsyx[-1]), np.sqrt(np.abs(eigvalsyx).min())

    self.L_y = 2.
    self.mu_y = 2.

    self.L = max(self.L_x, self.L_xy, self.L_y)

    self.grad_f = grad(self.f)
    self.grad_g = grad(self.g)

    self.xopt = xopt 
    self.yopt = yopt 
    self._proj_x = None
    self._proj_y = None
    # self.primal_func = self.dual_func = None
    
    N, M = self.A.shape
    self.gradx_complexity = 1+M
    self.grady_complexity = 1+N
    self.grad_f_complexity = 1
    self.grad_g_complexity = 1
    # self.prox_f_complexity = (1+K)**3 + 2
    # self.prox_g_complexity = (1+L)**3 + 2
    self.prox_f_complexity = 6
    self.prox_g_complexity = 4
    if self.xopt is not None and self.yopt is None:
        matrix = self.A.T
        vector = -self.grad_f(self.xopt)
        self.yopt = scipy.linalg.lstsq(matrix, vector, check_finite=True)[0]
        
    if self.xopt is None and self.yopt is None:
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=self.rho/2, fit_intercept=False)
        ridge.fit(self.A, self._y)
        self.xopt = ridge.coef_

        matrix = self.A.T
        vector = -self.grad_f(self.xopt)
        self.yopt = scipy.linalg.lstsq(matrix, vector, check_finite=True)[0]
    

  @classmethod
  def with_parameters(cls, nx, ny, L_x_mu_x, L_xy, mu_xy, L_y=None, mu_y=None):
    rvsx = lambda: ortho_group.rvs(dim=nx)
    rvsy = lambda: ortho_group.rvs(dim=ny)
    nA = min(nx, ny)
    eigvals_A = np.linspace(mu_xy, L_xy, nA)
    if nx < ny:
        mA = np.concatenate((np.diag(eigvals_A), np.zeros([nx, ny - nA])), axis=1)
    else:
        mA = np.concatenate((np.diag(eigvals_A), np.zeros([nx - nA, ny])), axis=0)
    A = rvsx().dot(mA).dot(rvsy()).T
    return cls(l=L_x_mu_x, A=A, y=np.random.randn(ny))

  def f(self, x):
    return self.rho / 2 * np.linalg.norm(x)**2

  def g(self, y):
    return (y - self._y).T @ (y - self._y)

  def F(self, x, y):
    return self.f(x) + y.T @ self.A @ x - self.g(y)

  def prox_f(self, v, scale=1.):
    return (1 - self.rho/2/max(v.T.dot(v), self.rho/2))*v

  def prox_g(self, v, scale=1.):
    return (1 - 1/max((v+2*self._y).T.dot((v+2*self._y)), 1))*(v+2*self._y) 

  def primal_func(self, x, y=None):
    """
    Computes the function value
    f_max(x) = \max_y F(x, y)

    F(x, y) = f(x) + <y, Ax> - g(y)
    f(x) = rho*0.5 <x, x>
    g(y) = <y - y_true, y - y_true>
    Args:
        x: np.array([dx])
    Returns:
        f_max(x): real function value
    """
    y_max = self._y + 0.5*self.A.dot(x)

    return self.F(x, y_max)

  def dual_func(self, y, x=None, func_lb=None):
    """
    Computes the function value
    g_min(y) = \min_x F(x, y)

    F(x, y) = f(x) + <y, Ax> - g(y)
    f(x) = rho*0.5 <x, x>
    g(y) = <y - y_true, y - y_true>
    Args:
        y: np.array([dy])
    Returns:
        h_min(y): real function value
    """
    x_min = -1/self.rho * self.A.T.dot(y)
    return self.F(x_min, y)

  def grad_f(self, x):
    return self.rho * x

  def grad_g(self, y):
    return 2*(y - self._y)

  def fg_grads(self, x, y):
    return self.grad_f(x), self.grad_g(y)

  def grad_A(self, x, y):
    grad_x = self.A.T.dot(y)
    grad_y = self.A.dot(x)

    return (grad_x, grad_y)

  def grad(self, x, y):
    dAdx, dAdy = self.grad_A(x, y)
    grad_x = self.grad_f(x) + dAdx
    grad_y = dAdy - self.grad_g(y)

    return (grad_x, grad_y)

  def loss(self, x, y):
    distance_squared = LA.norm(np.block([self.xopt]) - np.block([x]))

    return distance_squared
