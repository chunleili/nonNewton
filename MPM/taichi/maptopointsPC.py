import numpy as np
import scipy

from natcoords import natcoords
from shape3D import shape3D


def maptopointsPC(Ng, Tg, xmin, nexyz, Np, xp, vp, icon, vnew, v, dx, dt, p, Fr, g):
    # get natural coordinates of particles within cell
    [xpn, nep] = natcoords(xp, dx, xmin, nexyz)
    nep = nep.flatten()

    vnew = np.vstack([vnew[0:Ng], vnew[Ng : 2 * Ng], vnew[2 * Ng : 3 * Ng]]).T
    v = np.vstack([v[0:Ng], v[Ng : 2 * Ng], v[2 * Ng : 3 * Ng]]).T
    pp = np.zeros((Np))
    Tp = np.zeros((Np, 6))
    xpF = np.zeros((3, Np))
    k1 = np.zeros((3, Np))
    k2 = np.zeros((3, Np))
    k3 = np.zeros((3, Np))
    k4 = np.zeros((3, Np))

    ## RK4
    for ip in range(Np):
        i = nep[ip]
        iv = icon[:, i]
        n = shape3D(xpn[0, ip], xpn[1, ip], xpn[2, ip])

        k1[0, ip] = v[iv, 0].T @ n
        k1[1, ip] = v[iv, 1].T @ n
        k1[2, ip] = v[iv, 2].T @ n
        xpF[0, ip] = xp[0, ip] + k1[0, ip] * dt / 2
        xpF[1, ip] = xp[1, ip] + k1[1, ip] * dt / 2
        xpF[2, ip] = xp[2, ip] + k1[2, ip] * dt / 2
        [xpn2, nep2] = natcoords(xpF[:, ip].reshape(-1, 1), dx, xmin, nexyz)
        i2 = nep2[0]
        iv2 = icon[:, i2]
        nn = shape3D(xpn2[0], xpn2[1], xpn2[2])
        k2[0, ip] = 0.5 * (vnew[iv2, 0] + v[iv2, 0]).T @ nn
        k2[1, ip] = 0.5 * (vnew[iv2, 1] + v[iv2, 1]).T @ nn
        k2[2, ip] = 0.5 * (vnew[iv2, 2] + v[iv2, 2]).T @ nn

        xpF[0, ip] = xp[0, ip] + k2[0, ip] * dt / 2
        xpF[1, ip] = xp[1, ip] + k2[1, ip] * dt / 2
        xpF[2, ip] = xp[2, ip] + k2[2, ip] * dt / 2
        [xpn2, nep2] = natcoords(xpF[:, ip].reshape(-1, 1), dx, xmin, nexyz)
        i2 = nep2[0]
        iv2 = icon[:, i2]
        nn = shape3D(xpn2[0], xpn2[1], xpn2[2])
        k3[0, ip] = 0.5 * (vnew[iv2, 0] + v[iv2, 0]).T @ nn
        k3[1, ip] = 0.5 * (vnew[iv2, 1] + v[iv2, 1]).T @ nn
        k3[2, ip] = 0.5 * (vnew[iv2, 2] + v[iv2, 2]).T @ nn

        xpF[0, ip] = xp[0, ip] + k3[0, ip] * dt
        xpF[1, ip] = xp[1, ip] + k3[1, ip] * dt
        xpF[2, ip] = xp[2, ip] + k3[2, ip] * dt
        [xpn2, nep2] = natcoords(xpF[:, ip].reshape(-1, 1), dx, xmin, nexyz)
        i2 = nep2[0]
        iv2 = icon[:, i2]
        nn = shape3D(xpn2[0], xpn2[1], xpn2[2])
        k4[0, ip] = (vnew[iv2, 0]).T @ nn
        k4[1, ip] = (vnew[iv2, 1]).T @ nn
        k4[2, ip] = (vnew[iv2, 2]).T @ nn

        xp[0, ip] = xp[0, ip] + dt * (k1[0, ip] / 6 + k2[0, ip] / 3 + k3[0, ip] / 3 + k4[0, ip] / 6)
        xp[1, ip] = xp[1, ip] + dt * (k1[1, ip] / 6 + k2[1, ip] / 3 + k3[1, ip] / 3 + k4[1, ip] / 6)
        xp[2, ip] = (
            xp[2, ip]
            + dt * (k1[2, ip] / 6 + k2[2, ip] / 3 + k3[2, ip] / 3 + k4[2, ip] / 6)
            + 1 / 2 * g / Fr**2 * dt**2
        )

        vp[0, ip] = k1[0, ip] / 6 + k2[0, ip] / 3 + k3[0, ip] / 3 + k4[0, ip] / 6
        vp[1, ip] = k1[1, ip] / 6 + k2[1, ip] / 3 + k3[1, ip] / 3 + k4[1, ip] / 6
        vp[2, ip] = (k1[2, ip] / 6 + k2[2, ip] / 3 + k3[2, ip] / 3 + k4[2, ip] / 6) + g / Fr**2 * dt

        ### Update stress
        nPressure = p[iv]
        pp[ip] = nPressure.T @ n
        T1 = Tg[iv].T @ n
        T2 = Tg[iv + Ng].T @ n
        T3 = Tg[iv + 2 * Ng].T @ n
        T4 = Tg[iv + 3 * Ng].T @ n
        T5 = Tg[iv + 4 * Ng].T @ n
        T6 = Tg[iv + 5 * Ng].T @ n
        Tp[ip, :] = [T1, T2, T3, T4, T5, T6]
    return xp, vp, Tp, pp
