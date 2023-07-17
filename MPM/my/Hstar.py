import numpy as np
import scipy
import taichi as ti
from time import time


def Hstar_np(T, alf, gamma, Q0, xi, We, n, G1, vnew, Md):
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


@ti.kernel
def Hstar_kernel(
    T: ti.types.ndarray(),
    GV: ti.types.ndarray(),
    H: ti.types.ndarray(),
    f: ti.types.ndarray(),
    alf: ti.f32,
    gamma: ti.f32,
    Q0: ti.f32,
    xi: ti.f32,
    We: ti.f32,
    n: ti.i32,
):
    for i in range(n):
        Txx = T[i]
        Tyy = T[i + n]
        Tzz = T[i + 2 * n]
        Txy = T[i + 3 * n]
        Txz = T[i + 4 * n]
        Tyz = T[i + 5 * n]
        Tn = ti.Matrix([[Txx, Txy, Txz], [Txy, Tyy, Tyz], [Txz, Tyz, Tzz]])
        Gxx = GV[i]
        Gyy = GV[i + n]
        Gzz = GV[i + 2 * n]
        Gxy = GV[i + 3 * n]
        Gxz = GV[i + 4 * n]
        Gyz = GV[i + 5 * n]
        gv = ti.Matrix([[Gxx, Gxy, Gxz], [Gxy, Gyy, Gyz], [Gxz, Gyz, Gzz]])
        lam = ti.sqrt(1.0 + 1 / (3 * xi) * (Txx + Tyy + Tzz))
        ff = 1.0  # 2/gamma*(1-1/lam)*np.exp(Q0*(lam-1))+1/lam**2*(1-alf/(3*xi**2)*np.trace(Tn@Tn))
        hh = -1.0 / We * (xi * (ff - 1.0) * ti.Matrix.identity(ti.f32, 3) + alf / xi * (Tn @ Tn))
        gvt = gv @ Tn + Tn @ gv.transpose()
        rH = hh + gvt

        H[i] = rH[0, 0]
        H[i + n] = rH[1, 1]
        H[i + 2 * n] = rH[2, 2]
        H[i + 3 * n] = (rH[1, 0] + rH[0, 1]) / 2
        H[i + 4 * n] = (rH[2, 0] + rH[0, 2]) / 2
        H[i + 5 * n] = (rH[2, 1] + rH[1, 2]) / 2
        f[i] = ff


def Hstar_ti(T, alf, gamma, Q0, xi, We, n, G1, vnew, Md):
    # original
    A = Md.tocsr()
    b = G1 @ vnew
    GV = scipy.sparse.linalg.spsolve(A, b)

    # taichi variables
    GV_ti = ti.ndarray(shape=GV.shape, dtype=ti.f32)
    GV_ti.from_numpy(GV)
    H_ti = ti.ndarray(shape=(6 * n), dtype=ti.f32)
    f_ti = ti.ndarray(shape=(n), dtype=ti.f32)

    T_ti = ti.ndarray(shape=T.shape, dtype=ti.f32)
    T_ti.from_numpy(T)

    Hstar_kernel(T_ti, GV_ti, H_ti, f_ti, alf, gamma, Q0, xi, We, n)

    H = H_ti.to_numpy()
    f = f_ti.to_numpy()

    return H, f


def Hstar(T, alf, gamma, Q0, xi, We, n, G1, vnew, Md):
    H, f = Hstar_ti(T, alf, gamma, Q0, xi, We, n, G1, vnew, Md)
    return H, f
