import numpy as np
import scipy


def Matrix_reduction(nn, nk, Ng, Phi, vg, Hg, Tg, p):
    Nk = Phi.T @ nn
    znk = np.zeros((nk, Ng))
    vg = vg.flatten()
    vk = scipy.sparse.block_diag([Phi.T, Phi.T, Phi.T]) @ vg
    Hk = scipy.sparse.block_diag([Phi.T, Phi.T, Phi.T, Phi.T, Phi.T, Phi.T]) @ Hg
    Tk = scipy.sparse.block_diag([Phi.T, Phi.T, Phi.T, Phi.T, Phi.T, Phi.T]) @ Tg
    pk = Phi.T @ p
    return Nk, vk, Hk, Tk, pk, znk
