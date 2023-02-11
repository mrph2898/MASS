from typing import List, Optional
from autograd import grad, jacobian, elementwise_grad
import numpy as np
from scipy import linalg as LA
from scipy.linalg import pinv
import scipy.linalg as sla
import numpy.matlib as mt
from scipy.stats import ortho_group
import scipy


def generate_sym(size):
    # create a row vector of given size
    A = mt.rand(1, size)

    # create a symmetric matrix size * size
    symmA = A.T * A
    return symmA


def gen_cond(n, cond):
    """
    Parameters
    ----------
    n : Matrix size
    cond : Condition number
    Returns
    -------
    P : Return a n by n SPD matrix given a condition number
    """
    cond_P = cond     # Condition number
    log_cond_P = np.log(cond_P)
    exp_vec = np.arange(-log_cond_P/4., log_cond_P * (n)/(4 * (n - 1)), log_cond_P/(2.*(n-1)))
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = LA.qr((np.random.rand(n, n) - 5.) * 200)
    V, _ = LA.qr((np.random.rand(n, n) - 5.) * 200)
    P = U.dot(S).dot(V.T)
    P = P.dot(P.T)
    return P


def get_A_fixed(lambda_min, lambda_max, n):
    eigenvals = np.zeros(n)
    eigenvals[1:-1] = np.random.randint(low=lambda_min**2, 
                                        high=lambda_max**2,
                                        size=n - 2)
    eigenvals[0] = lambda_min**2
    eigenvals[-1] = lambda_max**2
    S = np.diag(eigenvals)

    Q = ortho_group.rvs(dim=n)
    return sla.sqrtm(Q.T @ S @ Q).real


class BaseSaddle(object):
    def __init__(self, A:np.ndarray, 
                 proj_x=None, proj_y=None):
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
        
        self.A = A
        self.mu_xy = None
        self.mu_yx = None
        self.L_xy = None
        self.F = lambda x, y: self.f(x) + y.T @ self.A @ x - self.g(y)
        self.grad_f = grad(self.f)
        self.grad_g = grad(self.g)
        self.dFdx = grad(self.F)
        self.dFdy = grad(self.F, 1)
        self._proj_x = proj_x
        self._proj_y = proj_y
        self.proj_x = lambda x: x if proj_x is None else proj_x(x)
        self.proj_y = lambda x: x if proj_y is None else proj_y(x)
        

    def grad(self, x, y):
        derivs = np.array([self.dFdx(x, y), self.dFdy(x, y)])
        return derivs[0], derivs[1]
    
    def fg_grads(self, x, y):
        derivs = np.array([self.grad_f(x), self.grad_g(y)])
        return derivs[0], derivs[1]
    
    def loss(self, x, y):
        return np.sqrt(LA.norm(x-self.xopt)**2 + LA.norm(y-self.yopt)**2)
    
    
class GeneralSaddle(BaseSaddle):
    def __init__(self, n, cond = 10, spd=False, bc=False):
        if spd:
            self.A = gen_cond(n, cond)
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
        self.mu_yx = spectrum.min() 
        self.mu_xy = sla.svd(self.A.dot(self.A.T))[1].min()
        
        self.f = lambda x: self.B.transpose() @ x + x.transpose() @ x
        self.mu_x = 2
        self.L_x = 2
        self.g = lambda y: self.C.transpose() @ y + y.transpose() @ y
        self.mu_y = 2
        self.L_y = 2

        F =  lambda x,y:  self.f(x) + y.transpose() @ self.A @ x - self.g(y)
        self.constraint = False   
        self.grad_f = grad(self.f)
        self.grad_g = grad(self.g)
        
        self.dFdx = grad(F)
        self.dFdy = grad(F, 1)   
    
    
    def grad(self, x, y):
        derivs = np.array([self.dFdx(x,y), self.dFdy(x,y)])
        return derivs[0], derivs[1]
    
    def fg_grads(self, x, y):
        return self.grad_f(x), self.grad_g(y)
    
    
    def loss(self, x, y):
        return np.sqrt(LA.norm(x-self.xopt)**2 + LA.norm(y-self.yopt)**2)
    
    
class func2(BaseSaddle):
    def __init__(self, A):
        super().__init__(A=A)
        self.xopt, self.yopt = 0., 0.   
        self.xrange = [-10, 10, .2]
        self.yrange = [-10, 10, .2]
        self.f = lambda x : x**2
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
        proj_x=None, proj_y=None
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
        # print(self.dx, self.dy)
  
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
        if abs(eigvalsx[0] - 0.) < 1e-8:
            self.mu_x = 0.
        
        eigvalsxy = LA.svdvals(self.A @ self.A.T)
        self.L_xy, self.mu_xy = np.sqrt(eigvalsxy[0]), np.sqrt(eigvalsxy[-1])
  
        eigvalsyx = LA.svdvals(self.A.T @ self.A)
        self.L_yx, self.mu_yx = np.sqrt(eigvalsyx[0]), np.sqrt(eigvalsyx[-1])
  
        eigvalsy = LA.eigvalsh(self.C)
        self.L_y, self.mu_y = eigvalsy[-1], eigvalsy[0]
        if abs(eigvalsy[0] - 0.) < 1e-8:
            self.mu_y = 0.
  
        self.L = max(self.L_x, self.L_xy, self.L_y)
  
        self.xopt = None
        self.yopt = None
        self._proj_x = proj_x
        self._proj_y = proj_y
        self.proj_x = lambda x: x if proj_x is None else proj_x(x)
        self.proj_y = lambda x: x if proj_y is None else proj_y(x)
        

    @classmethod
    def with_parameters(cls, nx, ny, L_x, mu_x, L_xy, mu_xy, L_y, mu_y):
        rvsx = lambda: ortho_group.rvs(dim=nx)
        rvsy = lambda: ortho_group.rvs(dim=ny)
  
        nA = min(nx, ny)
        eigvals_A = np.linspace(mu_xy, L_xy, nA)
        if nx < ny:
            # print(np.diag(eigvals_A).shape)
            # print(np.zeros([ny, nx - nA]).shape)
            mA = np.concatenate((np.diag(eigvals_A), np.zeros([nx, ny - nA])), axis=1)
            # print(mA.shape)
        else:
            # print(np.diag(eigvals_A).shape)
            # print(np.zeros([ny, nx - nA]).shape)
            mA = np.concatenate((np.diag(eigvals_A), np.zeros([nx - nA, ny])), axis=0)
            # print(mA.shape)
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
  
        return 0.5 * x.T @ self.B @ x + self.b.T @ x

    def g(self, y):
        """
        Computes the function value
        g(y) = 0.5 <y, Cy> + <c, y>
  
        Args:
          y: np.array([dy])
        Returns:
          g(y): real function value
        """
  
        return 0.5 * y.T @ self.C @ y + self.c.T @ y

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
  
        return self.F(x, y_max)

    def dual_func(self, y, x=None):
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
        return self.F(x_min, y)
    
    def prox_f(self, v, scale=1.):
        xopt = LA.solve(scale*self.B + np.eye(self.B.shape[0]), v - scale*self.b)
        return xopt
        
    def prox_g(self, v, scale=1.):
        yopt = LA.solve(scale*self.C + np.eye(self.C.shape[0]), v - scale*self.c)
        return yopt

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
