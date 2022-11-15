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

    self.l = l
    self.L_x = l
    self.mu_x = self.L_x

    self.n = A.shape[0]
    self.y = y

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
    return cls(l=L_x_mu_x, A=A, y=np.random.randint(0, 2, (ny)) * 2 - 1)

  def f(self, x):
    return self.l / 2 * np.linalg.norm(x)**2

  def g(self, y):
    ny = y * -self.y
    return 1 / self.n * (ny.T @ np.log(ny) + (1 - ny).T @ np.log(1 - ny))

  def F(self, x, y):
    return self.f(x) + y.T @ self.A @ x - self.g(y)

  # def grad_f(self, x):
  #   return self.l * x

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

  def proj_y(self, y):
    ny = y.copy()
    ny *= self.y
    ny = np.clip(ny, -1+1e-15, -1e-15)
    ny *= self.y
    return ny

  def loss(self, x, y):
    if self.xopt is None or self.yopt is None:
        from sklearn.linear_model import LogisticRegression
        log_reg = LogisticRegression(penalty='l2', C=1/(self.n * self.l), multi_class='ovr', fit_intercept=False)
        log_reg.fit(self.A * self.n, self.y)
        self.xopt = log_reg.coef_[0]

        matrix = self.A.T
        vector = -self.grad_f(self.xopt)
        self.yopt = scipy.linalg.lstsq(matrix, vector, check_finite=True)[0]

    distance_squared = LA.norm(np.block([self.xopt]) - np.block([x]))

    return distance_squared
