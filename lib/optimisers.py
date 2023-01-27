from typing import List, Optional
from autograd import grad, jacobian, elementwise_grad
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from tqdm import tqdm
import math
import scipy 
from scipy import linalg
from numpy import linalg as LA

import lib.anderson_acceleration as AA
from lib.problems import BaseSaddle


def simgd(problem: BaseSaddle,
          x0: np.ndarray, 
          y0: np.ndarray,
          max_iter: int,
          lr: float,
          k: int=0,
          verbose=1
         ):
    x, y = x0.copy(), y0.copy()
    loss = [problem.loss(x, y)]
    x_hist, y_hist = [x], [y]
    
    bar = range(max_iter)
    if verbose > 0:
        bar = tqdm(bar, desc="SimGD")
    
    for i in bar:
        gx, gy = problem.grad(x, y)
        x = x - lr * gx
        y = y + lr * gy
        lo = problem.loss(x, y)
        loss.append(lo)  
        x_hist.append(x)
        y_hist.append(y)
    return loss, x_hist, y_hist


def altgd(problem: BaseSaddle,
          x0: np.ndarray, 
          y0: np.ndarray,
          max_iter: int,
          lr: float,
          k: int=0,
          verbose=1
         ):
    x, y = x0.copy(), y0.copy()
    
    x_hist, y_hist = [x], [y]
    loss = [problem.loss(x, y)]
        
    bar = range(max_iter)
    if verbose > 0:
        bar = tqdm(bar, desc="AltGD")
    
    for i in bar:
        g_x, _ = problem.grad(x,y)
        x = x - lr * g_x
        _, g_y = problem.grad(x,y)
        y = y + lr * g_y
        x_hist.append(x)
        y_hist.append(y)
        loss.append(problem.loss(x, y))
    
    return loss, x_hist, y_hist


def avg(problem: BaseSaddle,
          x0: np.ndarray, 
          y0: np.ndarray,
          max_iter: int,
          lr: float,
          k: int=0,
          verbose=1
       ):
    x, y = x0.copy(), y0.copy()
    loss = [problem.loss(x, y)]
    xavg, yavg = x, y
    x_hist, y_hist = [xavg], [yavg]
    
    bar = range(max_iter)
    if verbose > 0:
        bar = tqdm(bar, desc="Avg")
    
    for i in bar:
        x = x - lr/np.sqrt(i+1)*(y)
        y = y + lr/np.sqrt(i+1)*(x)        
        xavg = xavg*(i+1)/(i+2) + x/(i+2)
        yavg = yavg*(i+1)/(i+2) + y/(i+2)        
        x_hist.append(xavg)
        y_hist.append(yavg)
        loss.append(problem.loss(xavg, yavg))
    return loss, x_hist, y_hist


def eg(problem: BaseSaddle,
       x0: np.ndarray, 
       y0: np.ndarray,
       max_iter: int,
       lr: float,
       k: int=0,
       verbose=1):
    x, y = x0.copy(), y0.copy()
    loss = [problem.loss(x, y)]
    x_hist, y_hist = [x], [y]
    
    bar = range(max_iter)
    if verbose > 0:
        bar = tqdm(bar, desc="EG")
    
    for i in bar:
        gx, gy = problem.grad(x, y)
        x_ = x - lr * gx
        y_ = y + lr * gy
        gx, gy = problem.grad(x_, y_)
        x = x - lr * gx
        y = y + lr * gy
        lo = problem.loss(x, y)
        loss.append(lo)  
        x_hist.append(x)
        y_hist.append(y)
    return loss, x_hist, y_hist


def omd(problem: BaseSaddle,
        x0: np.ndarray, 
        y0: np.ndarray,
        max_iter: int,
        lr: float,
        k: int=0,
        verbose=1):
    x, y = x0.copy(), y0.copy()
    loss = [problem.loss(x, y)]
    x_hist, y_hist = [x], [y]
    
    x_l, y_l = 0.5*x0, 0.5*y0
    g_xl, g_yl = problem.grad(x_l,y_l)
    bar = range(max_iter)
    if verbose > 0:
        bar = tqdm(bar, desc="OMD")
    
    for i in bar:
        g_x, g_y = problem.grad(x,y)
        x = x - 2 * lr * g_x + lr * g_xl
        y = y + 2 * lr * g_y - lr * g_yl
        g_xl, g_yl =  g_x, g_y
        lo = problem.loss(x, y)
        loss.append(lo)   
        x_hist.append(x)
        y_hist.append(y)
    return loss, x_hist, y_hist


gamma = 1e-26 # For 1d problems, you can also set gamma=0.
def altGDAAM(problem: BaseSaddle,
             x0: np.ndarray, 
             y0: np.ndarray,
             max_iter: int,
             lr: float,
             k: int=0,
             type2: bool=True,
             reg: float=1e-10,
             verbose=1
            ):
    '''
    Proposed Methods: alternating GDA with Anderson Acceleration with numpy
    '''
    x, y = x0, y0
    x_hist, y_hist = [x], [y]
    loss = [problem.loss(x, y)]
        
    fp = np.vstack((x, y))
    aa = AA.numpyAA(2, k, type2=type2, reg=reg)
    bar = range(max_iter)
    if verbose > 0:
        bar = tqdm(bar, desc="AltGDA-AM")
    
    for i in bar:
        fpprev = np.copy(fp)
        g_x, _ = problem.grad(x, y)
        x_ = (1 - gamma) * x - lr * g_x
        _, g_y = problem.grad(x_, y)
        y_ = y + lr * g_y
        fp = np.vstack((x_, y_))
        fp = aa.apply(fpprev, fp)
        x, y = fp[0],fp[1]
        lo = problem.loss(x, y)
        loss.append(lo)
        x_hist.append(x)
        y_hist.append(y)
    return loss, x_hist, y_hist




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


def APDG(problem,
         x0: np.ndarray,
         y0: np.ndarray,
         max_iter: int,
         params: dict=None,
         verbose=1
        ):
    """
    params: dict with
        theta
        tau_x, tau_y
        eta_x, eta_y
        alpha_x, alpha_y
        beta_x, beta_y
        sigma_x, sigma_y
    """
    x, y = x0.copy(), y0.copy()
    x_f, y_f = x0.copy(), y0.copy()
    y_prev = y0.copy() 
    A = problem.A
    if params is None:
        params = _get_apdg_params(problem)
    
    loss = [problem.loss(x, y)]
    x_hist, y_hist = [x], [y]
    
    bar = range(max_iter)
    if verbose > 0:
        bar = tqdm(bar, desc="APDG")
    
    for i in bar:
        y_m = y + params['theta'] * (y - y_prev)
        x_g = params['tau_x'] * x + (1 - params['tau_x']) * x_f
        y_g = params['tau_y'] * y + (1 - params['tau_y']) * y_f

        grad_x, grad_y = problem.fg_grads(x_g, y_g)
        x = (x + params['eta_x'] * params['alpha_x'] * (x_g - x) -
             params['eta_x'] * params['beta_x'] * (A.T.dot(A.dot(x) - grad_y)) - 
             params['eta_x'] * (grad_x + A.T.dot(y_m))
            )
        y = (y + params['eta_y'] * params['alpha_y'] * (y_g - y) - 
             params['eta_y'] * params['beta_y'] * (A.dot(A.T.dot(y) + grad_x)) -
             params['eta_y'] * (grad_y - A.dot(x))
            )

        x_f = x_g + params['sigma_x'] * (x - x_hist[-1])
        y_f = y_g + params['sigma_y'] * (y - y_hist[-1])
        
        y_prev = y_hist[-1]
        x_hist.append(x)
        y_hist.append(y)
        loss.append(problem.loss(x, y))
    return loss, x_hist, y_hist
