import numpy as np
import scipy.linalg as LA
import numpy.linalg as npla
import matplotlib.pyplot as plt
import os
from IPython.display import display, Latex

from lib.problems import BaseSaddle
import lib.optimisers as opt
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
         params, one_dim=False, k=5
        ):
    all_methods = {}
    
    if 'apdg' in params:
        loss, x, y = opt.APDG(problem=problem, x0=x0.copy(), y0=y0.copy(), max_iter=iteration, params=params['apdg'])
        all_methods["APDG"] = {"marker": 'g--',
                               "loss_hist": loss,
                               "x_hist": x,
                               "y_hist": y
                              }
    if 'altgd' in params:
        loss, x, y = opt.altgd(problem=problem, x0=x0.copy(), y0=y0.copy(), max_iter=iteration, lr=params['altgd'])
        all_methods["AltGD"] = {"marker": '--',
                               "loss_hist": loss,
                               "x_hist": x,
                               "y_hist": y
                              }
    if 'eg' in params:
        loss, x, y = opt.eg(problem=problem, x0=x0.copy(), y0=y0.copy(), max_iter=iteration, lr=params['eg'])
        all_methods["EG"] = {"marker": 'k-^',
                               "loss_hist": loss,
                               "x_hist": x,
                               "y_hist": y
                              }
    if "omd" in params:
        loss, x, y = opt.omd(problem=problem, x0=x0.copy(), y0=y0.copy(), max_iter=iteration, lr=params['omd'])
        all_methods["OMD"] = {"marker": 'c-*',
                               "loss_hist": loss,
                               "x_hist": x,
                               "y_hist": y
                              }
    if 'AA' in params:
        loss, x, y = opt.altGDAAM(problem=problem, x0=x0.copy(), y0=y0.copy(), max_iter=iteration, lr=params['AA'], k=k)
        all_methods["AltGDA-RAM"] = {"marker": 'b->',
                                     "loss_hist": loss,
                                     "x_hist": x,
                                     "y_hist": y
                                    }
    if 'sigmd' in params:
        loss, x, y = opt.simgd(problem=problem, x0=x0.copy(), y0=y0.copy(), max_iter=iteration, lr=params['simgd'])  
        all_methods["SimGD"] = {"marker": 'm-',
                                "loss_hist": loss,
                                "x_hist": x,
                                "y_hist": y
                              }
        
    loss, x, y = lpd.LiftedPrimalDual(problem, x0, y0, iteration + 1)
    all_methods["LPD"] = {"marker": 'r-d',
                          "loss_hist": loss,
                          "x_hist": x,
                          "y_hist": y
                          }
    # allloss[4], allxpath[4], allypath[4]= simGDAAM(problem, x0, y0, iteration, lr=lrset['AA'], k=k)   
    return all_methods
