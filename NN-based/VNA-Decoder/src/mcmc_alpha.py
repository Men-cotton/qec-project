import numpy as np
import random as rand
import copy

from numba import njit
from src.rotated_surface_model import RotSurCode, _apply_random_stabilizer

class Chain_alpha:
    def __init__(self, code, pz_tilde, alpha):
        self.code = code
        self.pz_tilde = pz_tilde
        self.alpha = alpha
        self.p_logical = 0
        self.flag = 0

    def update_chain_fast(self, iters):
        if isinstance(self.code, RotSurCode):
            self.code.qubit_matrix = _update_chain_fast_rotated(self.code.qubit_matrix, self.pz_tilde, self.alpha, iters)
        else:
            raise ValueError("Fast chain updates not available for this code")

@njit(cache=True)
def _update_chain_fast_rotated(qubit_matrix, pz_tilde, alpha, iters):

    for _ in range(iters):
        new_matrix, (dx, dy, dz) = _apply_random_stabilizer(qubit_matrix)

        p = pz_tilde**(dz + alpha*(dx + dy))
        if p > 1 or rand.random() < p:
            qubit_matrix = new_matrix

    return qubit_matrix
