import numpy as np
from shape3D import shape3D


def P2G(Ng, icon, xpn, nep, n_p, vp, pp, Tp):
    mv = np.zeros((Ng, 1))
    pv = np.zeros((Ng, 3))
    v = np.zeros((Ng, 3))
    Tv = np.zeros((Ng, 6))
    nn = np.zeros((Ng, 1), int)
    p = np.zeros((Ng, 1))
    #  map velocities to grid
    nep = nep.flatten()
    for ip in range(n_p):
        i = nep[ip]
        iv = icon[:, i]
        n = shape3D(xpn[0, ip], xpn[1, ip], xpn[2, ip])
        mv_ = mv[iv].flatten()
        mv_ = mv_ + n
        mv[iv] = mv_.reshape(-1, 1)
        pv[iv, :] = pv[iv, :] + np.vstack([(vp[0, ip] * n), vp[1, ip] * n, vp[2, ip] * n]).T
        Tv[iv, :] = (
            Tv[iv, :]
            + np.vstack([Tp[ip, 0] * n, Tp[ip, 1] * n, Tp[ip, 2] * n, Tp[ip, 3] * n, Tp[ip, 4] * n, Tp[ip, 5] * n]).T
        )
        nn[iv, 0] = 1
        a = pp[ip] * n
        b = p[iv, :].flatten() + a
        p[iv, 0] = b
        ...

    index1, _ = np.where(mv >= 1.0e-14)
    index0, _ = np.where(mv < 1.0e-14)
    index1 = index1.reshape(-1, 1)
    index0 = index0.reshape(-1, 1)
    mv1 = mv[index1].reshape(-1, 1)
    v[index1, 0] = pv[index1, 0] / mv1[:]
    v[index1, 1] = pv[index1, 1] / mv1[:]
    v[index1, 2] = pv[index1, 2] / mv1[:]

    Tv[index1, 0] = Tv[index1, 0] / mv1[:]
    Tv[index1, 1] = Tv[index1, 1] / mv1[:]
    Tv[index1, 2] = Tv[index1, 2] / mv1[:]
    Tv[index1, 3] = Tv[index1, 3] / mv1[:]
    Tv[index1, 4] = Tv[index1, 4] / mv1[:]
    Tv[index1, 5] = Tv[index1, 5] / mv1[:]

    index1 = index1.flatten()
    index0 = index0.flatten()
    p[index1] /= mv[index1]

    p[index0] = 0

    v[index0, 0] = 0
    v[index0, 1] = 0
    v[index0, 2] = 0

    Tv[index0, 0] = 0
    Tv[index0, 1] = 0
    Tv[index0, 2] = 0
    Tv[index0, 3] = 0
    Tv[index0, 4] = 0
    Tv[index0, 5] = 0

    v = np.hstack([v[:, 0], v[:, 1], v[:, 2]]).T
    T1 = np.hstack([Tv[:, 0], Tv[:, 1], Tv[:, 2], Tv[:, 3], Tv[:, 4], Tv[:, 5]]).T

    return mv, v, T1, p, nn
