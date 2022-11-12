
import importlib
import numpy as np


def LiftedPrimalDual(
  prob, x_0, y_0, iterations,
  log_freq=None, log_prefix=None, log_init=True, print_freq=None,):
  """
  Algorithm: Primal-Dual algorithm for separable
  smooth convex-concave minmax problems of the form
  g(x, y) = f(x) + <y, Ax> - h(y)
  [THO]: Lifted Primal-Dual Method for Bilinearly 
  Coupled Smooth Minimax Optimization
  Minimize:
  f_m(x) = \\min_x \\max_y F(x, y)
  Maximize:
  g_m(y) = \\min_x F(x, y)
  Solved as a convex-concave smooth-minimax problem
  Args:
    prob: FiniteMinimaxProb
    K: None or int
    z_0: None or np.array([m])
    stepsize: None or float
    log_freq: None or int
    log_prefix: None or str
    log_init: None or bool
    print_freq: None of int
 
  Returns:
    output: dict
  """
  x_k, y_k = x_0, y_0
  x_kminus1, y_kminus1 = x_k, y_k

  _grad_f = lambda x: prob.grad_f(x) - prob.mu_x*x
  _grad_h = lambda y: prob.grad_g(y) - prob.mu_y*y

  xp_k = _grad_f(x_0)
  yp_k = _grad_h(y_0)
  xp_kminus1, yp_kminus1 = xp_k, yp_k

  if log_prefix is None:
    log_prefix = ''

  kappa_x = (-1+prob.L_x/prob.mu_x)

  kappa_y = (-1+prob.L_y/prob.mu_y)

  if prob.mu_x != 0.0 and prob.mu_y != 0.0:
    gamma = 1 + (
      (kappa_x)**0.5 + 
      2*prob.L_xy/((prob.mu_x*prob.mu_y)**0.5) +
      (kappa_y)**0.5
      )**(-1.0)

    stepsize_x = (1/prob.mu_x)*(
      (kappa_x)**0.5 + 
      2*prob.L_xy/((prob.mu_x*prob.mu_y)**0.5)
    )**(-1.0)

    stepsize_y = (1/prob.mu_y)*(
      (kappa_y)**0.5 + 
      2*prob.L_xy/((prob.mu_x*prob.mu_y)**0.5)
    )**(-1.0)

    if kappa_x == 0.0:
      stepsize_xp = 0.0
    else:
      stepsize_xp = (
        (kappa_x)**0.5
      )**(-1.0)

    if kappa_y == 0.0:
      stepsize_yp = 0.0
    else:
      stepsize_yp = (
        (kappa_y)**0.5
      )**(-1.0)
      
    theta_x = theta_y = theta_xp = theta_yp = 1/gamma
  else:
    _stepsize_x = 1/2/(prob.L_x - prob.mu_x + 1.e-32)
    _stepsize_y = 1/2/(prob.L_y - prob.mu_y + 1.e-32)

    if prob.mu_x == 0.0 and prob.mu_y != 0.0:
      _stepsize_x = (1./_stepsize_x + 16*prob.L_xy/prob.mu_y)**-1.0
    if prob.mu_x != 0.0 and prob.mu_y == 0.0:
      _stepsize_y = (1./_stepsize_y + 16*prob.L_xy/prob.mu_x)**-1.0

  bx_k, by_k = x_k, y_k

  grad_bx_k, grad_by_k = _grad_f(bx_k), _grad_h(by_k)
  grad_bx_kminus1, grad_by_kminus1 = grad_bx_k, grad_by_k

  loss = []

  xk_list = []
  yk_list = []

  #metric_dict, metrics_string = prob.metrics(x_k, y_k)
  output = {}
  #for key, value in metric_dict.items():
  #  output[key] = []

  for k in range(iterations):
    z_k = (x_k, y_k)

    if prob.mu_x == 0.0 or prob.mu_y == 0.0:
      theta_x = theta_y = theta_xp = theta_yp = k/(k+1)
      stepsize_x = (1.0/(k+1)/_stepsize_x + k*prob.mu_x/2.0)**-1.0
      stepsize_y = (1.0/(k+1)/_stepsize_y + k*prob.mu_y/2.0)**-1.0

    tx_kplus1 = x_k + theta_x*(x_k - x_kminus1)
    ty_kplus1 = y_k + theta_y*(y_k - y_kminus1)

    tgrad_x_kplus1 = grad_bx_k + theta_xp*(grad_bx_k - grad_bx_kminus1)
    tgrad_y_kplus1 = grad_by_k + theta_yp*(grad_by_k - grad_by_kminus1)

    x_kplus1 = (x_k - stepsize_x*(prob.A.T.dot(ty_kplus1) + tgrad_x_kplus1))/(1.+stepsize_x*prob.mu_x)
    y_kplus1 = (y_k + stepsize_y*(prob.A.dot(tx_kplus1) - tgrad_y_kplus1))/(1.+stepsize_y*prob.mu_y)

    if k != 0 and (prob.mu_x == 0.0 or prob.mu_y == 0.0):
      stepsize_xp = 2.0/k
      stepsize_yp = 2.0/k

    if k == 0 and (prob.mu_x == 0.0 or prob.mu_y == 0.0):
      bx_kplus1 =  x_kplus1
      by_kplus1 =  y_kplus1
    else:
      bx_kplus1 =  (bx_k+ stepsize_xp*x_kplus1)/(1+stepsize_xp)
      by_kplus1 =  (by_k+ stepsize_yp*y_kplus1)/(1+stepsize_yp)

    grad_bx_kplus1 = _grad_f(bx_kplus1)
    grad_by_kplus1 = _grad_h(by_kplus1)

    loss.append(prob.loss(x_k, y_k))

    xk_list.append(x_k)
    yk_list.append(y_k)

#    if log_freq is not None and ((log_init and k==0) or k%log_freq == 0):


      #metric_dict, metrics_string = prob.metrics(x_k, y_k)
#
#       for key, value in metric_dict.items():
#         output[key].append(value)
      #
      # if print_freq is not None and (k == 0 or k%print_freq == 0):
      #   log_string = (
      #     '{}k={},x_k={},y_k={};{};'
      #     ''.format(
      #     log_prefix, k, x_k[:][:3], y_k[:][:3], metrics_string,
      #     ))
      #   print('{}'.format(log_string))

    # update iterates
    x_kminus1, x_k = x_k, x_kplus1
    y_kminus1, y_k = y_k, y_kplus1

    bx_k = bx_kplus1
    by_k = by_kplus1

    grad_bx_kminus1, grad_bx_k = grad_bx_k, grad_bx_kplus1
    grad_by_kminus1, grad_by_k = grad_by_k, grad_by_kplus1

  return loss, xk_list, yk_list
