import numpy as np
import scipy


def SolveAssemble(X20, Hk, dMu, Mt, A22d, NBS, beta, Re, G1, D, G, xi, nk, vk, dt):
    A11 = scipy.sparse.eye(6 * nk) - dt * beta / Re * G1 @ dMu @ D
    A12 = -dt * G1 @ dMu @ D
    A13 = dt * G1 @ dMu @ G
    A21 = -2 * xi * dt**2 * beta / Re * G1 @ dMu @ D
    A22 = Mt + A22d - 2 * xi * dt**2 * G1 @ dMu @ D
    A23 = 2 * xi * dt**2 * G1 @ dMu @ G
    A31 = -dt * beta / Re * G.T @ dMu @ D
    A32 = dt * G.T @ dMu @ D
    A33 = dt * G.T @ dMu @ G

    X20 = np.asarray(X20).reshape(-1)
    Hk = np.asarray(Hk).reshape(-1)
    vk = np.asarray(vk).reshape(-1)

    B1 = G1 @ vk
    B2 = Mt @ X20 + dt * Mt @ Hk + 2 * xi * dt * G1 @ vk
    B3 = G.T @ vk

    A33[NBS, :] = 0
    for j in range(len(NBS)):
        nbs = NBS[j]
        A33[nbs, nbs] = 1

    return A11, A12, A13, A21, A22, A23, A31, A32, A33, B1, B2, B3
