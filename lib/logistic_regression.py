from typing import List, Optional
from autograd import grad, jacobian, elementwise_grad
import autograd.numpy as np
from numpy import linalg as LA
from scipy.linalg import pinv
import scipy.linalg as sla
from scipy.stats import ortho_group
import scipy


class LogisticRegression:
  """
  Minimize x \\in \\reals^dx
  and
  Maximize y \\in \\reals^dy:
  :
  F(x, y) = f(x) + <y, Ax> - g(y)
  """
  def __init__(
    self, l, A, y, L_y = None
    ):

    # L_x =  - Lipschitz constant of grad_x w.r.t x
    # mu_x = - Strong convexity constant of grad_x w.r.t x
    # L_xy =  - Lipschitz constant of grad_x w.r.t y
    # mu_xy = - "Strong convexity" constant of grad_x w.r.t y
    # L_yx =  - Lipschitz constant of grad_y w.r.t x
    # mu_yx = - "Strong concavity" constant of grad_y w.r.t x
    # L_y = - Lipschitz constant of grad_y w.r.t y
    # mu_y = - Strong concavity constant of grad_y w.r.t y

    if L_y == None:
      L_y = 10

    self.rho = l
    self.L_x = l
    self.mu_x = self.L_x

    self.n = A.shape[0]
    self.y = y.astype(int)
    assert set(np.unique(self.y)) == {0, 1}, f"Labels should be 0 and 1, now they are {set(np.unique(self.y))}"

    self.A = A / self.n

    eigvalsxy = LA.eigvalsh(self.A @ self.A.T)
    self.L_xy, self.mu_xy = np.sqrt(eigvalsxy[-1]), np.sqrt(np.abs(eigvalsxy).min())

    eigvalsyx = LA.eigvalsh(self.A.T @ self.A)
    self.L_yx, self.mu_yx = np.sqrt(eigvalsyx[-1]), np.sqrt(np.abs(eigvalsyx).min())

    self.L_y = L_y
    self.mu_y = 4

    self.L = max(self.L_x, self.L_xy, self.L_y)

    self.grad_f = grad(self.f)
    self.grad_g = grad(self.g)

    self.xopt = None
    self.yopt = None
    if self.xopt is None or self.yopt is None:
        from sklearn.linear_model import LogisticRegression
        log_reg = LogisticRegression(penalty='l2', C=1/(self.n * self.rho), multi_class='ovr', fit_intercept=False)
        log_reg.fit(self.A * self.n, self.y)
        self.xopt = log_reg.coef_[0]

        matrix = self.A.T
        vector = -self.grad_f(self.xopt)
        self.yopt = scipy.linalg.lstsq(matrix, vector, check_finite=True)[0]
    self._proj_x = None
    self._proj_y = None
    
    N, M = self.A.shape
    self.gradx_complexity = 1+M
    self.grady_complexity = 1+1+3+1+1+3+1+N
    self.grad_f_complexity = 1
    self.grad_g_complexity = 1+1+3+1+1+3+1
    # self.prox_f_complexity = (1+K)**3 + 2
    # self.prox_g_complexity = (1+L)**3 + 2
    # self.prox_f_complexity = K + 2
    # self.prox_g_complexity = L + 2
    

  @classmethod
  def with_parameters(cls, nx, ny, L_x_mu_x, L_xy, mu_xy, L_y=None, mu_y=None):
    rvsx = lambda: ortho_group.rvs(dim=nx)
    rvsy = lambda: ortho_group.rvs(dim=ny)
    L_xy *= ny
    mu_xy *= ny
    nA = min(nx, ny)
    eigvals_A = np.linspace(mu_xy, L_xy, nA)
    if nx < ny:
        mA = np.concatenate((np.diag(eigvals_A), np.zeros([nx, ny - nA])), axis=1)
    else:
        mA = np.concatenate((np.diag(eigvals_A), np.zeros([nx - nA, ny])), axis=0)
    A = rvsx().dot(mA).dot(rvsy()).T
    return cls(l=L_x_mu_x, A=A, y=np.random.randint(0, 2, (ny)))

  def f(self, x):
    return self.rho / 2 * np.linalg.norm(x)**2

  def g(self, y):
    # ny = y * -self.y
    y = np.clip(1 / (1 + np.exp(-y)), 0., 1.)
    return 1 / self.n * (self.y.T @ np.log(y + 1e-12) +
                         (1 - self.y).T @ np.log(1 - y +1e-12))

  def F(self, x, y):
    return self.f(x) + y.T @ self.A @ x - self.g(y)

  def primal_func(self, x, y=None):
    """
    Computes the function value
    f_max(x) = \max_y F(x, y)

    F(x, y) = f(x) + <y, Ax> - g(y)
    f(x) = rho * 0.5 <x, x>
    g(y) = 1/n * (<y_true, log(sigma(y))> + <(1 - y_true), log(1 - sigma(y))>)
    Args:
        x: np.array([dx])
    Returns:
        f_max(x): real function value
    """
    
    # return self.F(x, self.y)
    # Unfair, but it's true
    return self.F(x, self.yopt)

  def dual_func(self, y, x=None, func_lb=None):
    """
    Computes the function value
    g_min(y) = \min_x F(x, y)

    F(x, y) = f(x) + <y, Ax> - g(y)
    f(x) = rho * 0.5 <x, x>
    g(y) = 1/n * (<y_true, log(sigma(y))> + <(1 - y_true), log(1 - sigma(y))>)
    Args:
        y: np.array([dy])
    Returns:
        h_min(y): real function value
    """
    x_min = -1/self.rho*self.A.T.dot(y)
    return self.F(x_min, y)

  # def grad_f(self, x):
  #   return self.rho * x

  # def grad_g(self, y):
  #   return grad_y

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


    def prox_f(self, v, scale=1.):
        xopt = self._inv_BI.dot(v - self._prox_scale*self.b)
        return xopt
    

  def proj_y(self, y):
    ny = y.copy()
    ny = np.clip(ny, 1e-15, -1e-15)
    return ny

  def loss(self, x, y):
    if self.xopt is None or self.yopt is None:
        from sklearn.linear_model import LogisticRegression
        log_reg = LogisticRegression(penalty='l2', C=1/(self.n * self.rho), multi_class='ovr', fit_intercept=False)
        log_reg.fit(self.A * self.n, self.y)
        self.xopt = log_reg.coef_[0]

        matrix = self.A.T
        vector = -self.grad_f(self.xopt)
        self.yopt = scipy.linalg.lstsq(matrix, vector, check_finite=True)[0]

    distance_squared = LA.norm(np.block([self.xopt]) - np.block([x]))

    return distance_squared
