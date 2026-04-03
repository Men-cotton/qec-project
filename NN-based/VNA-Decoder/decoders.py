import numpy as np
import copy
from math import log, exp
from multiprocessing import Pool
from src.mcmc import Chain
from src.mcmc_alpha import Chain_alpha

def EWD_droplet_alpha(chain, steps, alpha, onlyshortest):

    #chain.code.qubit_matrix = chain.code.apply_stabilizers_uniform()
    # All unique chains will be saved in samples
    all_seen = set()
    seen_chains = {}
    shortest = 1000000
    # Do the metropolis steps and add to samples if new chains are found
    for _ in range(steps):
        chain.update_chain_fast(5)
        key = hash(chain.code.qubit_matrix.tobytes())
        if key not in all_seen:
            all_seen.add(key)
            lengths = chain.code.chain_lengths()
            eff_len = lengths[2] + alpha * sum(lengths[0:2])
            if onlyshortest:
                if eff_len < shortest: # New shortest chain
                    shortest = eff_len
                    seen_chains = {}
                    seen_chains[key] = eff_len
                elif eff_len == shortest:
                    seen_chains[key] = eff_len
            else:
                seen_chains[key] = eff_len
    
    return seen_chains


def EWD_alpha(init_code, pz_tilde, alpha, steps, pz_tilde_sampling=None, onlyshortest=True):

    pz_tilde_sampling = pz_tilde_sampling if pz_tilde_sampling is not None else pz_tilde
    nbr_eq_classes = init_code.nbr_eq_classes
    # Create chain with p_sampling, this is allowed since N(n) is independent of p.
    eq_chains = [None] * nbr_eq_classes
    for eq in range(nbr_eq_classes):
        eq_chains[eq] = Chain_alpha(copy.deepcopy(init_code), pz_tilde_sampling, alpha)
        eq_chains[eq].code.qubit_matrix = eq_chains[eq].code.to_class(eq)

    # Z_E will be saved in eqdistr
    eqdistr = np.zeros(nbr_eq_classes)
    beta = - np.log(pz_tilde)

    for eq in range(nbr_eq_classes):
        # go to class eq and apply stabilizers
        chain = eq_chains[eq]

        out = EWD_droplet_alpha(chain, steps, alpha, onlyshortest)
        for eff_len in out.values():
            eqdistr[eq] += exp(-beta*eff_len)
        out.clear()

    return (np.divide(eqdistr, sum(eqdistr)) * 100)

if __name__ == '__main__':
    from src.rotated_surface_model import RotSurCode
    import time

    size = 5
    steps = 10 * size ** 5
    p_error = 0.10
    p_sampling = 0.30
    p_xyz = np.array([0.09, 0.01, 0.09])
    init_code = RotSurCode(size)
    
    pz_tilde_sampling = 0.25
    pz_tilde = 0.3
    alpha = 2
    
    tries = 1
    distrs = np.zeros((tries, init_code.nbr_eq_classes))

    for i in range(2):
        init_code.generate_random_error(p_error)

        # p_tilde = pz_tilde + 2*pz_tilde**alpha
        # p_z = pz_tilde*(1-p_tilde)
        # p_x = p_y = pz_tilde**alpha * (1-p_tilde)
        # init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
        
        ground_state = init_code.define_equivalence_class()
        print('Ground state:', ground_state)

        #init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
        #init_qubit = np.copy(init_code.qubit_matrix)

        #class_init = class_sorted_mwpm(init_code)
        #mwpm_init = regular_mwpm(init_code)

        print('################ Chain', i+1 , '###################')
        
        for i in range(tries):
            t0 = time.time()
            distrs[i] = EWD(copy.deepcopy(init_code), p_error=p_error, p_sampling=p_sampling, steps=steps, droplets=4, conv_mult=0)
            print('Try EWD       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'time:', time.time()-t0)
            t0 = time.time()
            distrs[i] = MCMC(copy.deepcopy(init_code), p=p_error)
            print('Try MCMC       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'time:', time.time()-t0)
