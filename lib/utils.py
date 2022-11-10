import numpy as np
import scipy.linalg as sla
import numpy.linalg as npla
import lib.optimisers as opt
import matplotlib.pyplot as plt
import os

import scipy.linalg as sla
from scipy.stats import ortho_group 


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
    return sla.sqrtm(Q.T @ S @ Q)


def plot(x, y, z, dz_dx, dz_dy, 
         loss, xpath, ypath,
         iteration, k, start, solution,
         ranges, figname,
         fig_dir=None, markevery= 10):
    x0, y0 = start
    xsol, ysol = solution
    xmin, xmax, xstep = ranges["x"]
    ymin, ymax, ystep = ranges["y"]
    loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8 = loss
    xpath1, xpath2, xpath3, xpath4, xpath5, xpath6, xpath7, xpath8= xpath
    ypath1, ypath2, ypath3, ypath4, ypath5, ypath6, ypath7, ypath8 = ypath
    fig, axlist = plt.subplots(1, 2, figsize=(14,5))
    ax1 = axlist[0]
    ax2 = axlist[1]    
    ax1.contourf(x, y, z, 5, cmap=plt.cm.gray)
    ax1.quiver(x, y, x - dz_dx, y - dz_dy, alpha=.5)
    ax1.plot(xpath7, ypath7,  'm-', linewidth=2, label='SimGDA',markevery=markevery)
    ax1.plot(xpath1, ypath1, 'g--', linewidth=2, label='APDG',markevery=markevery)
    ax1.plot(xpath2, ypath2, '--',linewidth=2, label='AltGD',markevery=markevery)
    ax1.plot(xpath3, ypath3, 'k-^', linewidth=2, label='EG',markevery=markevery)
    ax1.plot(xpath4, ypath4, 'c-*', linewidth=2, label='OMD',markevery=markevery)
    ax1.plot(xpath6, ypath6, 'b->', linewidth=2, label='SimGDA-RAM', markevery=markevery)
    ax1.plot(xpath5, ypath5, 'r-d', linewidth=2, label='AltGDA-RAM', markevery=markevery)
    x_init = ax1.scatter(x0, y0, marker='s', s=250, c='g',alpha=1,zorder=3, label='Start')
    x_sol = ax1.scatter(xsol, ysol, s=250, marker='*', color='violet', zorder=3, label='Optima')
    ax1.legend([x_init, x_sol],['Start','Optima'], markerscale=1, loc=4, fancybox=True, framealpha=1., fontsize=20)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')  
    ax1.set_xlim([xmin,xmax])
    ax1.set_ylim([ymin,ymax])
    
    plot_interval =1
    # ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss7[::plot_interval], 'm-', markevery=markevery, label='SimGDA')
    ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss1[::plot_interval], 'g--', markevery=markevery, label='APDG')
    ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss2[::plot_interval], '--', markevery=markevery, label='AltGD')
    # ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss3[::plot_interval], 'k-^', markevery=markevery, label='EG')
    # ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss4[::plot_interval],'c-*', markevery=markevery, label='OMD')
    # ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss5[::plot_interval], 'r-d', markevery=markevery, label='SimGDA-RAM')
    ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss6[::plot_interval], 'b->', markevery=markevery, label='AltGDA-RAM')
    ax2.set_xlabel('Iteration')
    ax2.set_ylim([1e-25,1e4])
    ax2.set_ylabel('Distance to optimal')
    axlist.flatten()[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True, framealpha=1., fontsize=20, markerscale=2)
    if fig_dir is not None:
        fig.savefig(os.path.join(fig_dir, figname), dpi=300, bbox_inches = 'tight', pad_inches = 0)
    else:
        fig.savefig(figname, dpi=300, bbox_inches = 'tight', pad_inches = 0)

        
def main(problem, iteration, x0, y0, params, k=5):
    allloss = [[] for _ in  range(8)]
    allxpath = [[] for _ in  range(8)]
    allypath = [[] for _ in  range(8)]
    allloss[0], allxpath[0], allypath[0] = opt.APDG(problem=problem, x0=x0, y0=y0, iter_num=iteration, params=params['apdg'])
    allloss[1], allxpath[1], allypath[1] = opt.altgd(problem, x0, y0, iteration, lr=params['altgd'])
    # allloss[2], allxpath[2], allypath[2] = eg(problem, x0, y0, iteration, lr=lrset['eg'])
    # allloss[3], allxpath[3], allypath[3] = omd(problem, x0, y0, iteration, lr=lrset['omd'])
    # allloss[4], allxpath[4], allypath[4]= simGDAAM(problem, x0, y0, iteration, lr=lrset['AA'], k=k)   
    allloss[5], allxpath[5], allypath[5]= opt.altGDAAM(problem, x0, y0, iteration, lr=params['AA'] ,k=k)   
    # allloss[6], allxpath[6], allypath[6]= simgd(problem, x0, y0, iteration, lr=lrset['simgd'])   
    return allloss, allxpath, allypath