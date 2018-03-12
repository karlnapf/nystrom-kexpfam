"""
Script to generate ground truth samples from the posterior over hyperparameters
of a GP classifier on the UCI Glass dataset, as used in
"Gradient-free Hamiltonian Monte Carlo with efficient kernel exponential families"

The script can be executed multiple times, and the resulting samples should be thinned
and merged manually.

The script is a stand-alone version of the one found in the Python repository
https://github.com/karlnapf/kernel_hmc

Requires Shogun and its Python interface, any release since 4.0,
compiled with Python interface, http://www.shogun.ml

NOTE: A set of (thinned) samples is included in the "data" folder.
"""
from kernel_hmc.mini_mcmc.mini_mcmc import mini_mcmc
from kernel_hmc.proposals.base import standard_sqrt_schedule
from kernel_hmc.proposals.metropolis import AdaptiveMetropolis
from kernel_hmc.tools.log import Log
import numpy as np
import os
import datetime

logger = Log.get_logger()

from kernel_hmc.densities.posterior_gp_classification_ard import GlassPosterior

def get_am_instance(target):
    # adaptive version that tunes itself towards the "optimal" acceptance rate
    # set schedule=None for completely non-adaptive version
    step_size = 1.
    gamma2 = 0.1
    schedule = standard_sqrt_schedule
    acc_star = 0.234
    am = AdaptiveMetropolis(target, D, step_size, gamma2, schedule, acc_star)
    
    return am

if __name__ == '__main__':
    # Glass posterior has 9 dimensions
    D = 9
    target = GlassPosterior()
    target.set_up()

    # transition kernel, pick any
    sampler = get_am_instance(target)
        
    # MCMC parameters
    start = np.zeros(D)
    num_iter = 30000
    
    # run MCMC
    samples, proposals, accepted, acc_prob, log_pdf, times, step_sizes = mini_mcmc(sampler, start, num_iter, D)
    
    homedir = os.path.expanduser("~")
    timestamp = datetime.datetime.now().isoformat()
    fname = "gp_glass_sketch_D=%d_N=%d-%s.npz" % (D, num_iter, timestamp)
    with open(fname, "w") as f:
        np.savez(f,
                 samples=samples,
                 proposals=proposals,
                 accepted=accepted,
                 acc_prob=acc_prob,
                 log_pdf=log_pdf,
                 times=times,
                 step_sizes=step_sizes)
