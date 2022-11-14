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
    

def plot(x, y, z, dz_dx, dz_dy, 
         all_methods: dict,
         iteration, k, start, solution,
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
        ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval),
                     all_methods[method]["loss_hist"][::plot_interval],
                     all_methods[method]["marker"],
                     markevery=markevery,
                     label=method
                    )
    # ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss5[::plot_interval], 'r-d', markevery=markevery, label='SimGDA-RAM')
    ax2.set_xlabel('Iteration')
    ax2.set_ylim([1e-25,1e4])
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
    all_methods = {}
    
    if 'apdg' in params:
        # loss, x, y = opt.APDG(problem=problem, x0=x0.copy(), y0=y0.copy(), 
        #                       max_iter=iteration, params=params['apdg'], verbose=verbose)
        apdg_cls = copt.APDG(problem=problem, x0=x0.copy(), y0=y0.copy(), 
                             eps=eps, stopping_criteria='loss',
                             params=params['apdg'])
        loss, x, y = apdg_cls(max_iter=iteration,
                              verbose=verbose)
        all_methods["APDG"] = {"marker": 'g--',
                               "loss_hist": loss,
                               "x_hist": x,
                               "y_hist": y
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
        all_methods["AltGD"] = {"marker": '--',
                               "loss_hist": loss,
                               "x_hist": x,
                               "y_hist": y
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
        all_methods["EG"] = {"marker": 'k-^',
                               "loss_hist": loss,
                               "x_hist": x,
                               "y_hist": y
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
        all_methods["OMD"] = {"marker": 'c-*',
                               "loss_hist": loss,
                               "x_hist": x,
                               "y_hist": y
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
            all_methods["AltGDA-AM"] = {"marker": 'b->',
                                     "loss_hist": loss,
                                     "x_hist": x,
                                     "y_hist": y
                                    }
        except ValueError:
            all_methods["AltGDA-AM"] = {"marker": 'b->',
                                     "loss_hist": [np.nan],
                                     "x_hist": [np.nan],
                                     "y_hist": [np.nan]
                                    }
            
        
    if 'simgd' in params:
        simgd_cls = copt.SimGD(problem=problem, x0=x0.copy(), y0=y0.copy(), 
                       eps=eps, stopping_criteria='loss',
                       params=params['simgd']
                      )
    
        loss, x, y = simgd_cls(max_iter=iteration,
                               verbose=verbose)
        # loss, x, y = opt.simgd(problem=problem, x0=x0.copy(), y0=y0.copy(),
        #                        max_iter=iteration, lr=params['simgd'], verbose=verbose)  
        all_methods["SimGD"] = {"marker": 'm-',
                                "loss_hist": loss,
                                "x_hist": x,
                                "y_hist": y
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
        all_methods["AVG"] = {"marker": 'y-h',
                                "loss_hist": loss,
                                "x_hist": x,
                                "y_hist": y
                              }
        
    lpd_cls = copt.LPD(problem=problem, x0=x0.copy(), y0=y0.copy(), 
                       eps=eps, stopping_criteria='loss',
                       params=params['lpd']
                      )
    
    loss, x, y = lpd_cls(max_iter=iteration,
                         verbose=verbose)
    # loss, x, y = lpd.LiftedPrimalDual(problem, x0, y0, iteration + 1, verbose=verbose)
    all_methods["LPD"] = {"marker": 'r-d',
                          "loss_hist": loss,
                          "x_hist": x,
                          "y_hist": y
                          }
    # allloss[4], allxpath[4], allypath[4]= simGDAAM(problem, x0, y0, iteration, lr=lrset['AA'], k=k)   
    return all_methods
