import numpy as np
import math
import scipy 
from scipy import linalg
from numpy import linalg as LA
# from pyblas.level1 import dnrm2
from scipy.linalg.blas import dnrm2


class numpyAA:
    def __init__(self, dimension, depth, type2=True,reg=1e-7):
        self._dimension = dimension
        self._depth = depth
        self.xTx = None
        self.reg = reg
        self.Y = np.zeros((self._dimension, self._depth)) # changes in increments
        self.S = np.zeros((self._dimension, self._depth)) # changes in fixed point applications
        self.xTx = np.zeros((self._depth, self._depth))
        self.it = 0
        if type2:
            self.apply = self.type2
        else:
            self.apply = self.type1
        
    def reset(self):
        self.Y = np.zeros((self._dimension, self._depth)) # changes in increments
        self.S = np.zeros((self._dimension, self._depth)) # changes in fixed point applications
        self.xTx = np.zeros((self._depth, self._depth))
        self.it = 0
        
        
    def type2(self, x : np.ndarray, fx : np.ndarray) -> np.ndarray:

        g = x[:,0] -fx[:,0]
        mk = min(self.it, self._depth)
        if (self.it > 0):
            # Build matrices of changes
            col = (self.it -1) % self._depth
            y = g - self.gprev
            self.S[:,col] = x[:,0] - self.xprev[:,0]
            self.Y[:,col] = y
            A = self.Y[:,0:mk].transpose() @ self.Y[:,0:mk]
            b = self.Y[:,0:mk].transpose()@ g
            normS = dnrm2(self.S[:,0:mk], self._dimension, incx=1)
            normY = dnrm2(self.Y[:,0:mk], self._dimension, incx=1)
            reg = normS**2 + normY**2
#             try:
#                 res =  scipy.linalg.lapack.dgesv(A + self.reg * reg * np.eye(mk), b)
#                 gamma_k = res[2]
#             except scipy.linalg.LinAlgError as e:
#                 if 'Singular matrix' in str(e):
            lstsq_solution = linalg.lstsq(A + self.reg  * reg * np.eye(mk), b)
            gamma_k = lstsq_solution[0]
            xkp1 = fx - np.dot(self.S[:,0:mk] - self.Y[:,0:mk], gamma_k)[:, np.newaxis]
            self.it +=1
            self.xprev = x.copy()
            self.gprev = x[:,0] - fx[:,0]

        else:
            xkp1 = fx
            self.it +=1
            self.xprev = x.copy()
            self.gprev = x[:,0] - fx[:,0]
        
        return xkp1
    
    def type1(self, x : np.ndarray, fx : np.ndarray) -> np.ndarray:
        
        
        g = x[:,0] -fx[:,0]
        mk = min(self.it, self._depth)
        if (self.it > 0):
            # Build matrices of changes
            col = (self.it -1) % self._depth
            s = x[:,0] - self.xprev[:,0]
            self.S[:,col] = s
            self.Y[:,col] = g - self.gprev
            A = self.S[:,0:mk].transpose() @ self.Y[:,0:mk]
            b = self.S[:,0:mk].transpose()@ g
            normS = dnrm2(self.S[:,0:mk], self._dimension, incx=1)
            normY = dnrm2(self.Y[:,0:mk], self._dimension, incx=1)
            reg = normS**2 + normY**2
            lstsq_solution = linalg.lstsq(A + self.reg * reg * np.eye(mk), b)
            gamma_k = lstsq_solution[0]
            xkp1 = fx - np.dot(self.S[:,0:mk] - self.Y[:,0:mk], gamma_k)[:, np.newaxis]
            self.it +=1
            self.xprev = x.copy()
            self.gprev = x[:,0] - fx[:,0]
            
        else:
            xkp1 = fx
            self.it +=1
            self.xprev = x.copy()
            self.gprev = x[:,0] - fx[:,0]
        
        return xkp1