from typing import List, Optional
from autograd import grad, jacobian, elementwise_grad
import numpy as np
from numpy import linalg as LA
from scipy.linalg import pinv
import scipy.linalg as sla
from scipy.stats import ortho_group
import scipy
import numpy.matlib as mt
import lib.utils as ut


class BaseSaddle(object):
    def __init__(self, A:np.ndarray):
        self.xopt = None
        self.yopt = None
        self.xrange = None
        self.yrange = None
        self.f = None
        self.mu_x = None
        self.L_x = None
        
        self.g = None 
        self.mu_y = None
        self.L_y = None
        
        self.A = None
        self.mu_xy = None
        self.mu_yx = None
        self.L_xy = None
        self.F = lambda x, y: self.f(x) + y.T @ self.A @ x - self.g(y)
        self.grad_f = grad(self.f)

        self.grad_g = grad(self.g)
        self.dFdx = grad(self.F)
        self.dFdy = grad(self.F, 1)
        # self.d2fdxdx = grad(self.dfdx)
        # self.d2fdydy = 0
        # self.d2fdxdy = grad(self.dfdx, 1)
        # self.d2fdydx = grad(self.dfdy)

    def fr(self, x, y):
        "this is used for the baseline model(follow the ridge)"
        yy = self.d2fdydy(x, y)
        yx = self.d2fdydx(x, y)
        if yy == 0:
            return 0
        return yx/yy
    
    def grad(self, x, y):
        return self.dFdx(x, y), self.dFdy(x, y)

    def fg_grads(self, x, y):
        return self.grad_f(x), self.grad_g(y)
    
    def loss(self, x, y):
        return np.sqrt(LA.norm(x-self.xopt)**2 + LA.norm(y-self.yopt)**2)
    
    
class GeneralSaddle(BaseSaddle):
    def __init__(self, n, cond = 10, spd=False, bc=False):
        if spd:
            self.A = ut.gen_cond(n, cond)
        else:
            self.A = np.random.randn(n,n)
        if bc:
            print('bc')
            self.B = np.random.randn(n,1)
            self.C = np.random.randn(n,1)
            self.xopt = LA.solve(self.A, -self.C)
            self.yopt = LA.solve(self.A.transpose(), -self.B)    
        else:
            print('bc zeros')
            self.B = np.zeros((n,1))
            self.C = np.zeros((n,1))
            self.xopt = LA.solve(self.A, -self.C)
            self.yopt = LA.solve(self.A.transpose(), -self.B) 
            
        spectrum = sla.svd(self.A.T.dot(self.A))[1]
        self.L_xy = spectrum.max()**.5
        self.mu_yx = spectrum.min()**.5
        self.mu_xy = sla.svd(self.A.dot(self.A.T))[1].min()

        self.f = lambda x: x.transpose() @ x + self.B.transpose() @ x
        self.mu_x = 2
        self.L_x = 2
        self.g = lambda y: y.transpose() @ y + self.C.transpose() @ y
        self.mu_y = 2
        self.L_y = 2
        self.F = lambda x, y: self.f(x) + y.T @ self.A @ x - self.g(y)

        self.constraint = False
        self.grad_f = grad(self.f)
        self.grad_g = grad(self.g)

        self.dFdx = grad(self.F)
        self.dFdy = grad(self.F, 1)
    
    
    def fr(self, x, y):
        yx = self.d2Fdydx(x, y)
        return yx

    def grad(self, x, y):
        derivs = np.array([self.dFdx(x,y), self.dFdy(x,y)])
        return derivs[0], derivs[1]

    def fg_grads(self, x, y):
        derivs = np.array([self.grad_f(x), self.grad_g(y)])
        return derivs[0], derivs[1]


    def loss(self, x, y):
        return np.sqrt(LA.norm(x-self.xopt)**2 + LA.norm(y-self.yopt)**2)

    
    
class func2(BaseSaddle):
    def __init__(self, A):
        super().__init__(A=A)
        self.xopt, self.yopt = 0., 0.   
        self.xrange = [-10, 10, .2]
        self.yrange = [-10, 10, .2]
        self.f = lambda x: x**2
        self.mu_x = 2
        self.L_x = 2
        self.g = lambda y: y**2
        self.mu_y = 2
        self.L_y = 2
        self.A = A
        spectrum = sla.svd(A.T.dot(A))[1]
        self.L_xy = spectrum.max()**.5
        self.mu_yx = spectrum.min()**.5
        self.mu_xy = sla.svd(A.dot(A.T))[1].min()**.5
        
        self.constraint = False   
        self.grad_f = grad(self.f)
        self.grad_g = grad(self.g)

class BilinearQuadraticSaddle:
  """
  Minimize x \\in \\reals^dx
  and
  Maximize y \\in \\reals^dy:
  :
  F(x, y) = f(x) + <y, Ax> - g(y)
  f(x) = 0.5 <x, Bx> + <b, x>
  h(y) = 0.5 <y, Cy> + <c, y>
  primal(x) = \\max_{y} F(x, y)
  dual(x) = \\min_{x} F(x, y)
  Attributes:
    dx: dimension of the x variable
    dy: dimension of the y variable
    A: np.array([dy, dx])
    B: np.array([dx, dx])
    b: np.array([dx])
    C: np.array([dy, dy])
    c: np.array([dy])
  """
  def __init__(
    self, A, B=None, C=None, b=None, c=None,
    Dx=None, Dy=None, dual_func=None,
    ):
    """
    Args:
      A: np.array([dy, dx])
      B: None or np.array([dx, dx])
      b: None or np.array([dx])
      C: None or np.array([dy, dy])
      c: None or np.array([dy])
    """
    self.A = np.array(A)
    self.dy, self.dx = self.A.shape
    print(self.dx, self.dy)

    if B is not None:
      self.B = np.array(B)
    else:
      self.B = np.zeros([self.dx, self.dx])
    self.B = (self.B + self.B.T)/2

    if b is not None:
      self.b = np.array(b)
    else:
      self.b = np.zeros([self.dx])

    if C is not None:
      self.C = np.array(C)
    else:
      self.C = np.zeros([self.dy, self.dy])
    self.C = (self.C + self.C.T)/2

    if c is not None:
      self.c = np.array(c)
    else:
      self.c = np.zeros([self.dy])

    # L_x =  - Lipschitz constant of grad_x w.r.t x
    # mu_x = - Strong convexity constant of grad_x w.r.t x
    # L_xy =  - Lipschitz constant of grad_x w.r.t y
    # mu_xy = - "Strong convexity" constant of grad_x w.r.t y
    # L_yx =  - Lipschitz constant of grad_y w.r.t x
    # mu_yx = - "Strong concavity" constant of grad_y w.r.t x
    # L_y = - Lipschitz constant of grad_y w.r.t y
    # mu_y = - Strong concavity constant of grad_y w.r.t y

    eigvalsx = LA.eigvalsh(self.B)
    self.L_x, self.mu_x = eigvalsx[-1], eigvalsx[0]

    eigvalsxy = LA.eigvalsh(self.A @ self.A.T)
    self.L_xy, self.mu_xy = np.sqrt(eigvalsxy[-1]), np.sqrt(eigvalsxy[0])

    eigvalsyx = LA.eigvalsh(self.A.T @ self.A)
    self.L_yx, self.mu_yx = np.sqrt(eigvalsyx[-1]), np.sqrt(eigvalsyx[0])

    eigvalsy = LA.eigvalsh(self.C)
    self.L_y, self.mu_y = eigvalsy[-1], eigvalsy[0]

    self.L = max(self.L_x, self.L_xy, self.L_y)

    self.xopt = None
    self.yopt = None

  @classmethod
  def with_parameters(cls, nx, ny, L_x, mu_x, L_xy, mu_xy, L_y, mu_y):
    rvsx = lambda: ortho_group.rvs(dim=nx)
    rvsy = lambda: ortho_group.rvs(dim=ny)

    nA = min(nx, ny)
    eigvals_A = np.linspace(mu_xy, L_xy, nA)
    if nx < ny:
        print(np.diag(eigvals_A).shape)
        print(np.zeros([ny, nx - nA]).shape)
        mA = np.concatenate((np.diag(eigvals_A), np.zeros([nx, ny - nA])), axis=1)
        print(mA.shape)
    else:
        print(np.diag(eigvals_A).shape)
        print(np.zeros([ny, nx - nA]).shape)
        mA = np.concatenate((np.diag(eigvals_A), np.zeros([nx - nA, ny])), axis=0)
        print(mA.shape)
    A = rvsx().dot(mA).dot(rvsy()).T

    eigvals_B = np.linspace(mu_x**0.5, L_x**0.5, nx)
    B = rvsx().dot(np.diag(eigvals_B).dot(rvsx()))
    B = B.T.dot(B)

    eigvals_C = np.linspace(mu_y**0.5, L_y**0.5, ny)
    C = rvsy().dot(np.diag(eigvals_C).dot(rvsy()))
    C = C.T.dot(C)

    return cls(A=A, B=B, C=C)


  def f(self, x):
    """
    Computes the function value
    f(x) = 0.5 <x, Bx> + <b, x>

    Args:
      x: np.array([dx])
    Returns:
      f(x): real function value
    """

    return 0.5 * x.T @ B @ x + b.T @ x

  def g(self, y):
    """
    Computes the function value
    g(y) = 0.5 <y, Cy> + <c, y>

    Args:
      y: np.array([dy])
    Returns:
      g(y): real function value
    """

    return 0.5 * y.T @ self.C @ y + c.T @ y

  def F(self, x, y):
    """
    Computes the function value
    F(x, y) = f(x) + <y, Ax> - g(y)
    f(x) = 0.5 <x, Bx> + <b, x>
    g(y) = 0.5 <y, Cy> + <c, y>

    Args:
      (x, y): (np.array([dx]), np.array([dy]))
    Returns:
      F(x, y): real function value
    """
    return self.f(x) + y.T @ self.A @ x - self.g(y)

  def grad_f(self, x):
    """
    Computes the gradient of the function
    f(x) = 0.5 <x, Bx> + <b, x>
    grad_x = grad_x f(x)
    Returns:
      grad f(x):grad_x gradient
    """
    grad_x = self.B.dot(x) + self.b

    return grad_x

  def grad_g(self, y):
    """
    Computes the gradient of the function
    g(y) = 0.5 <y, Cy> + <c, y>
    grad_y = grad_g h(y)
    Returns:
      grad g(y):grad_y gradient
    """
    grad_y = self.C.dot(y) + self.c

    return grad_y

  def fg_grads(self, x, y):
    return self.grad_f(x), self.grad_g(y)

  def grad_A(self, x, y):
    """
    Computes the gradient of the function
    <y, Ax>
    (grad_x, grad_y)
    grad_x = grad_x <y, Ax>
    grad_y = grad_y <y, Ax>
    Returns:
      grad F(x, y):(grad_x, grad_y) gradient
    """

    grad_x = self.A.T.dot(y)
    grad_y = self.A.dot(x)

    return (grad_x, grad_y)

  def grad(self, x, y):
    """
    Computes the gradient of the function
    F(x, y) = f(x) + <y, Ax> - g(y)
    f(x) = 0.5 <x, Bx> + <b, x>
    g(y) = 0.5 <y, Cy> + <c, y>
    """
    dAdx, dAdy = self.grad_A(x, y)
    grad_x = self.grad_f(x) + dAdx
    grad_y = dAdy - self.grad_g(y)

    return (grad_x, grad_y)

  def primal_func(self, x, y=None):
    """
    Computes the function value
    f_max(x) = \max_y F(x, y)

    F(x, y) = f(x) + <y, Ax> - g(y)
    f(x) = 0.5 <x, Bx> + <b, x>
    g(y) = 0.5 <y, Cy> + <c, y>
    Args:
      x: np.array([dx])
    Returns:
      f_max(x): real function value
    """
    matrix = self.C
    vector = self.A.dot(x) - self.c
    try:
      y_max = scipy.linalg.solve(matrix, vector, check_finite=True, assume_a='sym')
    except scipy.linalg.LinAlgError:
      y_max = scipy.linalg.lstsq(matrix, vector, check_finite=True)[0]

    return self.func((x, y_max))

  def dual_func(self, y, x=None, func_lb=None):
    """
    Computes the function value
    g_min(y) = \min_x F(x, y)

    F(x, y) = f(x) + <y, Ax> - g(y)
    f(x) = 0.5 <x, Bx> + <b, x>
    g(y) = 0.5 <y, Cy> + <c, y>
    Args:
      y: np.array([dy])
    Returns:
      h_min(y): real function value
    """
    matrix = self.B
    vector = -self.A.T.dot(y) - self.b
    try:
      x_min = scipy.linalg.solve(matrix, vector, check_finite=True, assume_a='sym')
    except scipy.linalg.LinAlgError:
      x_min = scipy.linalg.lstsq(matrix, vector, check_finite=True)[0]
    return self.func((x_min, y))

  def loss(self, x, y):
    if self.mu_x != 0.0 and self.mu_y != 0.0:
      if self.xopt is None or self.yopt is None:
        matrix = np.block([[self.B, self.A.T], [-self.A, self.C]])
        vector = np.block([-self.b,-self.c])
        try:
          opt = scipy.linalg.solve(matrix, vector, check_finite=True, assume_a='gen')
        except scipy.linalg.LinAlgError:
          opt = scipy.linalg.lstsq(matrix, vector, check_finite=True)[0]
        self.xopt, self.yopt = opt[:self.dx], opt[self.dx:]

      distance_squared = LA.norm(np.block([self.xopt, self.yopt]) - np.block([x, y]))
    else:
      distance_squared = np.inf

    return distance_squared

