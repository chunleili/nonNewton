import numpy as np
import scipy


def full_scale_operators(dxm, dym, dzm, Ck, Mf, Phi, Re, beta, zk):
    # Gradient of Velocity
    l1 = scipy.sparse.hstack([dxm, zk, zk])
    l2 = scipy.sparse.hstack([zk, dym, zk])
    l3 = scipy.sparse.hstack([zk, zk, dzm])
    l4 = scipy.sparse.hstack([0.5 * dym, 0.5 * dxm, zk])
    l5 = scipy.sparse.hstack([0.5 * dzm, zk, 0.5 * dxm])
    l6 = scipy.sparse.hstack([zk, 0.5 * dzm, 0.5 * dym])
    G1 = scipy.sparse.vstack([l1, l2, l3, l4, l5, l6])
    # Divergence of stress
    l1 = scipy.sparse.hstack([dxm, zk, zk, dym, dzm, zk])
    l2 = scipy.sparse.hstack([zk, dym, zk, dzm, zk, dxm])
    l3 = scipy.sparse.hstack([zk, zk, dzm, zk, dxm, dym])
    D = scipy.sparse.vstack([l1, l2, l3])
    # Gradient of P
    G = scipy.sparse.vstack([dxm, dym, dzm])
    # Mass matrix
    Mu = scipy.sparse.block_diag([Ck, Ck, Ck])
    Md = scipy.sparse.block_diag([Ck, Ck, Ck, Ck, Ck, Ck])
    Mt = Md
    Mu = np.sum(Mu, axis=1)
    Mu_inv = 1.0 / Mu
    Mu_inv = np.asarray(Mu_inv).reshape(-1)
    dMu = scipy.sparse.diags(Mu_inv)
    Md = Re / (2 * beta) * np.sum(Md, axis=1)
    Mt = np.sum(Mt, axis=1)
    Mt = np.asarray(Mt).reshape(-1)
    Mt = scipy.sparse.diags(Mt)
    Mf = Phi.T @ Mf @ Phi
    A22d = scipy.sparse.block_diag([Mf, Mf, Mf, Mf, Mf, Mf])
    return G1, D, G, Mu, Md, Mt, dMu, A22d
