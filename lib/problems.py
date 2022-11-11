from typing import List, Optional
from autograd import grad, jacobian, elementwise_grad
import numpy as np
from numpy import linalg as LA
from scipy.linalg import pinv
import scipy.linalg as sla
import numpy.matlib as mt


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
        self.F = lambda x, y: self.f(x, y) + x.T @ self.A @ y - self.g(x, y)
        self.dfdx = grad(self.f)
        self.dfdy = 0
        
        self.dgdx = 0
        self.dgdy = grad(self.g, 1) 
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
        derivs = np.array([self.dFdx(x, y), self.dFdy(x, y)])
        return derivs[0], derivs[1]
    
    def fg_grads(self, x, y):
        derivs = np.array([self.dfdx(x, y), self.dgdy(x, y)])
        return derivs[0], derivs[1]
    
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
            self.xopt = LA.solve(self.A.transpose(), -self.C)
            self.yopt = LA.solve(self.A, -self.B)    
        else:
            print('bc zeros')
            self.B = np.zeros((n,1))
            self.C = np.zeros((n,1))
            self.xopt = LA.solve(self.A.transpose(), -self.C)
            self.yopt = LA.solve(self.A, -self.B) 
            
        spectrum = sla.svd(self.A.T.dot(self.A))[1]
        self.L_xy = spectrum.max()**.5
        self.mu_yx = spectrum.min() 
        self.mu_xy = sla.svd(self.A.dot(self.A.T))[1].min()
        
        self.f = lambda x, y : x.transpose() @ x
        self.mu_x = 2
        self.L_x = 2
        self.g = lambda x, y: y.transpose() @ y
        self.mu_y = 2
        self.L_y = 2

        F =  lambda x,y:  self.f(x, y) - self.g(x, y) + x.transpose() @ self.A @ y + self.B.transpose() @ x + self.C.transpose() @ y 
        self.constraint = False   
        self.dfdx = grad(self.f)
        self.dfdy = 0
        self.dgdx = 0
        self.dgdy = grad(self.g, 1) 
        
        self.dFdx = grad(F)
        self.dFdy = grad(F, 1)   
        self.d2Fdxdx = grad(self.dFdx)
        self.d2Fdydy = grad(self.dFdy, 1)
        self.d2Fdxdy = grad(self.dFdx, 1)
        self.d2Fdydx = grad(self.dFdy)
    
    
    def fr(self, x, y):
        yx = self.d2Fdydx(x, y)
        return yx
    
    def grad(self, x, y):
        derivs = np.array([self.dFdx(x,y), self.dFdy(x,y)])
        return derivs[0], derivs[1]
    
    def fg_grads(self, x, y):
        derivs = np.array([self.dfdx(x,y), self.dgdy(x,y)])
        return derivs[0], derivs[1]
    
    
    def loss(self, x, y):
        return np.sqrt(LA.norm(x-self.xopt)**2 + LA.norm(y-self.yopt)**2)

    
    
class func2(BaseSaddle):
    def __init__(self, A):
        super().__init__(A=A)
        self.xopt, self.yopt = 0., 0.   
        self.xrange = [-10, 10, .2]
        self.yrange = [-10, 10, .2]
        self.f = lambda x, y : x**2
        self.mu_x = 2
        self.L_x = 2
        self.g = lambda x, y: y**2
        self.mu_y = 2
        self.L_y = 2
        self.A = A
        spectrum = sla.svd(A.T.dot(A))[1]
        self.L_xy = spectrum.max()**.5
        self.mu_yx = spectrum.min()**.5
        self.mu_xy = sla.svd(A.dot(A.T))[1].min()**.5
        
        self.constraint = False   
        self.dfdx = grad(self.f)  
        self.dgdy = grad(self.g, 1)