import numpy as np
import scipy


def full_scale_operators(dxm, dym, dzm, Ck, Mf, Phi, Re, beta, zk):
    # Gradient of Velocity
    G1 = scipy.sparse.bmat(
        [
            [dxm, zk, zk],
            [zk, dym, zk],
            [zk, zk, dzm],
            [0.5 * dym, 0.5 * dxm, zk],
            [0.5 * dzm, zk, 0.5 * dxm],
            [zk, 0.5 * dzm, 0.5 * dym],
        ]
    )
    # Divergence of stress
    D = scipy.sparse.bmat([[dxm, zk, zk, dym, dzm, zk], [zk, dym, zk, dzm, zk, dxm], [zk, zk, dzm, zk, dxm, dym]])
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
