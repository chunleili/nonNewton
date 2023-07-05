import numpy as np
import scipy


def Hstar(T, alf, gamma, Q0, xi, We, n, G1, vnew, Md):
    A = Md
    b = (G1 @ vnew).reshape(-1, 1)
    GV = scipy.sparse.linalg.spsolve(A, b)
    H = np.zeros((6 * n, 1))
    f = np.zeros((n, 1))
    for i in range(n):
        Txx = T[i]
        Tyy = T[i + n]
        Tzz = T[i + 2 * n]
        Txy = T[i + 3 * n]
        Txz = T[i + 4 * n]
        Tyz = T[i + 5 * n]
        Tn = np.vstack((np.hstack((Txx, Txy, Txz)), np.hstack((Txy, Tyy, Tyz)), np.hstack((Txz, Tyz, Tzz))))
        Gxx = GV[i]
        Gyy = GV[i + n]
        Gzz = GV[i + 2 * n]
        Gxy = GV[i + 3 * n]
        Gxz = GV[i + 4 * n]
        Gyz = GV[i + 5 * n]
        gv = np.vstack((np.hstack((Gxx, Gxy, Gxz)), np.hstack((Gxy, Gyy, Gyz)), np.hstack((Gxz, Gyz, Gzz))))
        lam = np.sqrt(1 + 1 / (3 * xi) * (np.trace(Tn)))
        ff = 1  # 2/gamma*(1-1/lam)*np.exp(Q0*(lam-1))+1/lam**2*(1-alf/(3*xi**2)*np.trace(Tn@Tn))
        hh = -1 / We * (xi * (ff - 1) * np.eye(3) + alf / xi * (Tn @ Tn))
        gvt = gv @ Tn + Tn @ gv.T
        rH = hh + gvt

        H[i, 0] = rH[0, 0]
        H[i + n, 0] = rH[1, 1]
        H[i + 2 * n, 0] = rH[2, 2]
        H[i + 3 * n, 0] = (rH[1, 0] + rH[0, 1]) / 2
        H[i + 4 * n, 0] = (rH[2, 0] + rH[0, 2]) / 2
        H[i + 5 * n, 0] = (rH[2, 1] + rH[1, 2]) / 2
        f[i, 0] = ff

    return H, f
