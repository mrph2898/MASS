import numpy as np
import scipy.linalg as LA
import numpy.linalg as npla
import matplotlib.pyplot as plt
import os
from IPython.display import display, Latex

from lib.problems import BaseSaddle
import lib.optimisers as opt
import lib.cls_optimisers as copt
import lib.lpd as lpd


def display_constants(problem: BaseSaddle):
    display(Latex(r"$L_{xy}$ = " + f"{problem.L_xy:.5f}," +
                  r"$\mu_{xy}$ = " + f"{problem.mu_xy:.5f}," +
                  r"$\mu_{yx}$ = " + f"{problem.mu_yx:.5f}" 
                 ))
    display(Latex(r"$L_{x}$ = " + f"{problem.L_x:.5f}," +
                  r"$\mu_{x}$ = " + f"{problem.mu_x:.5f}" 
                 ))
    display(Latex(r"$L_{y}$ = " + f"{problem.L_y:.5f}," +
                  r"$\mu_{y}$ = " + f"{problem.mu_y:.5f}" 
                 ))
    display(Latex(r"$\sqrt{\frac{L_x}{\mu_{x}}}$ = " + f"{(problem.L_x / problem.mu_x)**.5:.5f}" + 
                  r"$\sqrt{\frac{L_y}{\mu_{y}}}$ = " + f"{(problem.L_y / problem.mu_y)**.5:.5f}" +
                  r"$\frac{L_{xy}}{\sqrt{\mu_{x} \mu_y}}$ = " + f"{problem.L_xy / (problem.mu_x*problem.mu_y)**.5:.5f}"))
    display(Latex(r"$\frac{\sqrt{L_x L_y}}{\mu_{xy}}$ = " + f"{(problem.L_x * problem.L_y)**.5 / problem.mu_xy:.5f}" +
                  r"$\frac{L_{xy}}{\mu_{xy}}\cdot\sqrt{\frac{L_x}{\mu_{x}}}$ = " + f"{(problem.L_xy / problem.mu_xy)*(problem.L_x / problem.mu_x)**.5:.5f}" + 
                  r"$\frac{L_{xy}^2}{\mu_{xy}^2}$ = " + f"{(problem.L_xy**2 / problem.mu_xy**2):.5f}"))
    display(Latex(r"$\frac{\sqrt{L_x L_y}}{\mu_{yx}}$ = " + f"{(problem.L_x * problem.L_y)**.5 / problem.mu_yx:.5f}" +
                  r"$\frac{L_{xy}}{\mu_{yx}}\cdot\sqrt{\frac{L_y}{\mu_{y}}}$ = " + f"{(problem.L_xy / problem.mu_yx)*(problem.L_y / problem.mu_y)**.5:.5f}"  +
                  r"$\frac{L_{xy}^2}{\mu_{yx}^2}$ = " + f"{(problem.L_xy**2 / problem.mu_yx**2):.5f}"
                 ))
    display(Latex(r"$\frac{\sqrt{L_x L_y} L_{xy}}{\mu_{xy}\mu_{yx}}$ = " + f"{(problem.L_x * problem.L_y)**.5 / problem.mu_xy:.5f}" +
                  r"$\frac{L_{xy}^2}{\mu_{yx}^2}$ = " + f"{(problem.L_xy**2 / problem.mu_yx**2):.5f}" + 
                  r"$\frac{L_{xy}^2}{\mu_{xy}^2}$ = " + f"{(problem.L_xy**2 / problem.mu_xy**2):.5f}"
                 ))
    params = opt._get_apdg_params(problem)
    _pstr = ""
    for name, val in params.items():
        _pstr += f"$\\{name}$ = {val:.6f}\n"
    display(Latex(_pstr))
    
    
def metrics(problem, x, y):
    """
        The function has been taken from 
        https://github.com/tkkiran/LiftedPrimalDual/blob/20696d51113551d0fe965d0f15ba904bb01f65c0/minmax/probs.py#L22
    """
    metrics_dict = {}
    z = (x, y)

    if problem.primal_func is not None and problem.dual_func is not None:
        p_func = problem.primal_func(x, y=y)
        d_func = problem.dual_func(y, x=x)
        gap = p_func - d_func
        try:
            assert gap >= 0 or np.isclose(gap, 0.)
        except:
            print('p_func={}, d_func={},gap={}'.format(p_func, d_func, gap))
            raise ValueError('Gap is negative!')
        metrics_dict['gap'] = gap
    
    grad_x, grad_y = problem.grad(x, y)
    if problem._proj_x is not None:
        x = z[0]
        stepsize_x = 1/problem.L/10
        grad_x = (x - problem.proj_x(x - stepsize_x*grad_x))/stepsize_x
    if problem._proj_y is not None:
        y = z[1]
        stepsize_y = 1/problem.L/10
        grad_y = (y - problem.proj_y(y - stepsize_y*grad_y))/stepsize_y

    metrics_dict['grad_norm'] = ((grad_x**2).sum() + (grad_y**2).sum())**0.5
    metrics_dict['func'] = problem.F(x, y)

    metrics_string = ""
    if problem.primal_func is not None and problem.dual_func is not None:
        metrics_string += f"gap={metrics_dict['gap']:2.2g}"
    metrics_string += ",|grad|={metrics_dict['grad_norm']:2.2g},func={metrics_dict['func']:.2g}"
    return metrics_dict, metrics_string
    

def plot(x, y, z, dz_dx, dz_dy, 
         all_methods: dict,
         iteration, eps, start, solution,
         ranges, figname,
         fig_dir=None, markevery= 10):
    x0, y0 = start
    xsol, ysol = solution
    xmin, xmax, xstep = ranges["x"]
    ymin, ymax, ystep = ranges["y"]
    fig, axlist = plt.subplots(1, 2, figsize=(14,5))
    ax1 = axlist[0]
    ax2 = axlist[1]    
    ax1.contourf(x, y, z, 5, cmap=plt.cm.gray)
    ax1.quiver(x, y, x - dz_dx, y - dz_dy, alpha=.5)
    
    for method in all_methods:
        ax1.plot(all_methods[method]["x_hist"], 
                 all_methods[method]["y_hist"], 
                 all_methods[method]["marker"],
                 label=method,
                 markevery=markevery
                )
    # ax1.plot(xpath6, ypath6, 'r-d', linewidth=2, label='SimGDA-RAM', markevery=markevery) 
    x_init = ax1.scatter(x0, y0, marker='s', s=250, c='g',alpha=1,zorder=3, label='Start')
    x_sol = ax1.scatter(xsol, ysol, s=250, marker='*', color='violet', zorder=3, label='Optima')
    ax1.legend([x_init, x_sol],['Start','Optima'], markerscale=1, loc=4, fancybox=True, framealpha=1., fontsize=20)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')  
    ax1.set_xlim([xmin,xmax])
    ax1.set_ylim([ymin,ymax])
    
    plot_interval = 1
    for method in all_methods:
        ax2.semilogy(np.arange(0, len(all_methods[method]["loss_hist"])+plot_interval-1, plot_interval),
                     all_methods[method]["loss_hist"][::plot_interval],
                     all_methods[method]["marker"],
                     markevery=markevery,
                     label=method
                    )
    # ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss5[::plot_interval], 'r-d', markevery=markevery, label='SimGDA-RAM')
    ax2.set_xlabel('Iteration')
    ax2.set_ylim([eps - eps/2, 1e4])
    ax2.set_ylabel('Distance to optimal')
    axlist.flatten()[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True, framealpha=1., fontsize=20, markerscale=2)
    if fig_dir is not None:
        fig.savefig(os.path.join(fig_dir, figname), dpi=300, bbox_inches = 'tight', pad_inches = 0)
    else:
        fig.savefig(figname, dpi=300, bbox_inches = 'tight', pad_inches = 0)

        
        
def main(problem, iteration, 
         x0, y0,
         params,
         eps=1e-9,
         verbose=1
        ):
    
    # TODO: Drop histories from result dict (because class has already it)
    all_methods = {}

    if 'apdg' in params:
        # loss, x, y = opt.APDG(problem=problem, x0=x0.copy(), y0=y0.copy(), 
        #                       max_iter=iteration, params=params['apdg'], verbose=verbose)
        apdg_cls = copt.APDG(problem=problem, x0=x0.copy(), y0=y0.copy(), 
                             eps=eps, stopping_criteria='loss',
                             params=params['apdg'])
        loss, x, y = apdg_cls(max_iter=iteration,
                              verbose=verbose)
        all_methods["APDG"] = {
            "class": apdg_cls,
            "marker": 'g-1',
            "loss_hist": loss,
            "x_hist": x,
            "y_hist": y,
            "iters_spent": apdg_cls.iter_count,
            "total_time": apdg_cls.time
        }
        
    if 'altgd' in params:
        altgd_cls = copt.AltGD(problem=problem, x0=x0.copy(), y0=y0.copy(), 
                       eps=eps, stopping_criteria='loss',
                       params=params['altgd']
                      )
    
        loss, x, y = altgd_cls(max_iter=iteration,
                               verbose=verbose)
        # loss, x, y = opt.altgd(problem=problem, x0=x0.copy(), y0=y0.copy(), 
        #                        max_iter=iteration, lr=params['altgd'], verbose=verbose)
        all_methods["AltGD"] = {
            "class": altgd_cls,
            "marker": '--p',
            "loss_hist": loss,
            "x_hist": x,
            "y_hist": y,
            "iters_spent": altgd_cls.iter_count,
            "total_time": altgd_cls.time
        }
        
    if 'smm' in params:
        smm_cls = copt.SepMiniMax(problem=problem, x0=x0.copy(), y0=y0.copy(), 
                       eps=eps, stopping_criteria='loss',
                       params=params['smm']
                      )
    
        loss, x, y = smm_cls(max_iter=iteration,
                               verbose=verbose)
        # loss, x, y = opt.altgd(problem=problem, x0=x0.copy(), y0=y0.copy(), 
        #                        max_iter=iteration, lr=params['altgd'], verbose=verbose)
        all_methods["SeparateMiniMax"] = {
            "class": smm_cls,
            "marker": '-v',
            "loss_hist": loss,
            "x_hist": x,
            "y_hist": y,
            "iters_spent": smm_cls.iter_count,
            "total_time": smm_cls.time
        }
    
    if 'foam' in params:
        foam_cls = copt.FOAM(problem=problem, x0=x0.copy(), y0=y0.copy(), 
                       eps=eps, stopping_criteria='loss',
                       params=params['foam']
                      )
    
        loss, x, y = foam_cls(max_iter=iteration,
                               verbose=verbose)
        # loss, x, y = opt.altgd(problem=problem, x0=x0.copy(), y0=y0.copy(), 
        #                        max_iter=iteration, lr=params['altgd'], verbose=verbose)
        all_methods["FOAM"] = {
            "class": foam_cls,
            "marker": '-X',
            "loss_hist": loss,
            "x_hist": x,
            "y_hist": y,
            "iters_spent": foam_cls.iter_count,
            "total_time": foam_cls.time
        }
        
    if "acceg" in params:
        _params = copt.FOAM._get_foam_params(problem)
        _alpha = _params["alpha"]
        _theta = _params["theta"]
        _mu_y = problem.mu_y
        
        _inner_iter = int(max(1/_alpha, _alpha/(_theta*_mu_y))*np.log(1/eps))
        inner_optimiser = copt.FOAM

        acceg_cls = copt.AccEG(problem,
                          inner_optimiser=inner_optimiser,
                          inner_max_iter=_inner_iter,
                          x0=x0.copy(), 
                          y0=y0.copy(),
                          eps=eps,
                          stopping_criteria="loss",
                          params=params["acceg"]
                         )
        loss, x, y = acceg_cls(max_iter=iteration,
                               verbose=verbose)
        all_methods["AccEG"] = {
            "class": acceg_cls,
            "marker": '-^',
            "loss_hist": loss,
            "x_hist": x,
            "y_hist": y,
            "iters_spent": acceg_cls.iter_count,
            "total_time": acceg_cls.time
        }
        
    if "accel-eg" in params:
        _params = copt.FOAM._get_foam_params(problem)
        _alpha = _params["alpha"]
        _theta = _params["theta"]
        _mu_y = problem.mu_y
        
        _inner_iter = int(max(1/_alpha, _alpha/(_theta*_mu_y))*np.log(1/eps))
        inner_optimiser = copt.LPD

        acc_eg_cls = copt.AcceleratedEG(problem,
                          inner_optimiser=inner_optimiser,
                          inner_max_iter=_inner_iter,
                          x0=x0.copy(), 
                          y0=y0.copy(),
                          eps=eps,
                          stopping_criteria="loss",
                          params=params["accel-eg"]
                         )
        loss, x, y = acc_eg_cls(max_iter=iteration,
                               verbose=verbose)
        all_methods["AcceleratedEG"] = {
            "class": acc_eg_cls,
            "marker": '-^',
            "loss_hist": loss,
            "x_hist": x,
            "y_hist": y,
            "iters_spent": acc_eg_cls.iter_count,
            "total_time": acc_eg_cls.time
        }

    if "twice-aceg" in params:
        twice_acc_eg_cls = copt.TwiceAcceleratedEG(problem,
                                             x0=x0.copy(),
                                             y0=y0.copy(),
                                             eps=eps,
                                             stopping_criteria="loss",
                                             params=params["twice-aceg"]
                                        )
        loss, x, y = twice_acc_eg_cls(max_iter=iteration,
                                verbose=verbose)
        all_methods["TwiceAcceleratedEG"] = {
            "class": twice_acc_eg_cls,
            "marker": '-^',
            "loss_hist": loss,
            "x_hist": x,
            "y_hist": y,
            "iters_spent": twice_acc_eg_cls.iter_count,
            "total_time": twice_acc_eg_cls.time
        }
        
    if 'eg' in params:
        eg_cls = copt.EG(problem=problem, x0=x0.copy(), y0=y0.copy(), 
                       eps=eps, stopping_criteria='loss',
                       params=params['eg']
                      )
    
        loss, x, y = eg_cls(max_iter=iteration,
                            verbose=verbose)
        # loss, x, y = opt.eg(problem=problem, x0=x0.copy(), y0=y0.copy(), 
        #                     max_iter=iteration, lr=params['eg'], verbose=verbose)
        all_methods["EG"] = {
            "class": eg_cls,
            "marker": 'k-^',
            "loss_hist": loss,
            "x_hist": x,
            "y_hist": y,
            "iters_spent": eg_cls.iter_count,
            "total_time": eg_cls.time
        }
        
    if "omd" in params:
        omd_cls = copt.OMD(problem=problem, x0=x0.copy(), y0=y0.copy(), 
                       eps=eps, stopping_criteria='loss',
                       params=params['omd']
                      )
    
        loss, x, y = omd_cls(max_iter=iteration,
                             verbose=verbose)
        # loss, x, y = opt.omd(problem=problem, x0=x0.copy(), y0=y0.copy(), 
        #                      max_iter=iteration, lr=params['omd'], verbose=verbose)
        all_methods["OMD"] = {
            "class": omd_cls,
            "marker": 'c--*',
            "loss_hist": loss,
            "x_hist": x,
            "y_hist": y,
            "iters_spent": omd_cls.iter_count,
            "total_time": omd_cls.time
        }
        
    if 'AA' in params:
        try:
            altgdaam_cls = copt.AltGDAAM(problem=problem, x0=x0.copy(), y0=y0.copy(), 
                       eps=eps, stopping_criteria='loss',
                       params=params['AA']
                      )
    
            loss, x, y = altgdaam_cls(max_iter=iteration,
                                 verbose=verbose)
            # loss, x, y = opt.altGDAAM(problem=problem, x0=x0.copy(), y0=y0.copy(),
            #                           max_iter=iteration, lr=params['AA'], k=k, verbose=verbose)
            all_methods["AltGDA-AM"] = {
                "class": altgdaam_cls,
                "marker": 'b-->',
                "loss_hist": loss,
                "x_hist": x,
                "y_hist": y,
                "iters_spent": altgdaam_cls.iter_count,
                "total_time": altgdaam_cls.time
            }
        except ValueError:
            print("AltGDA-AM couldn't be used in such parameters' settings")

    if 'simgd' in params:
        simgd_cls = copt.SimGD(problem=problem, x0=x0.copy(), y0=y0.copy(), 
                       eps=eps, stopping_criteria='loss',
                       params=params['simgd']
                      )
    
        loss, x, y = simgd_cls(max_iter=iteration,
                               verbose=verbose)
        # loss, x, y = opt.simgd(problem=problem, x0=x0.copy(), y0=y0.copy(),
        #                        max_iter=iteration, lr=params['simgd'], verbose=verbose)  
        all_methods["SimGD"] = {
            "class": simgd_cls,
            "marker": 'm--o',
            "loss_hist": loss,
            "x_hist": x,
            "y_hist": y,
            "iters_spent": simgd_cls.iter_count,
            "total_time": simgd_cls.time
        }
        
    if 'avg' in params:
        avg_cls = copt.Avg(problem=problem, x0=x0.copy(), y0=y0.copy(), 
                       eps=eps, stopping_criteria='loss',
                       params=params['avg']
                      )
    
        loss, x, y = avg_cls(max_iter=iteration,
                             verbose=verbose)
        # loss, x, y = opt.avg(problem=problem, x0=x0.copy(), y0=y0.copy(),
        #                        max_iter=iteration, lr=params['avg'], verbose=verbose)  
        all_methods["AVG"] = {
            "class": avg_cls,
            "marker": 'y--h',
            "loss_hist": loss,
            "x_hist": x,
            "y_hist": y,
            "iters_spent": avg_cls.iter_count,
            "total_time": avg_cls.time
        }
        
    if 'lpd' in params:
        lpd_cls = copt.LPD(problem=problem, x0=x0.copy(), y0=y0.copy(), 
                           eps=eps, stopping_criteria='loss',
                           params=params['lpd']
                          )

        loss, x, y = lpd_cls(max_iter=iteration,
                             verbose=verbose)
        # loss, x, y = lpd.LiftedPrimalDual(problem, x0, y0, iteration + 1, verbose=verbose)
        all_methods["LPD"] = {
            "class": lpd_cls,
            "marker": 'r-d',
            "loss_hist": loss,
            "x_hist": x,
            "y_hist": y,
            "iters_spent": lpd_cls.iter_count,
            "total_time": lpd_cls.time
        }
        
    return all_methods
