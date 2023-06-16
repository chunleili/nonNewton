import numpy as np
from shape3D import shape3D
def InitialParticle(npe, dx, icon, Xg, Ne):
    wt = 2.0 / npe[:]
    xp = []
    vp = []
    for i in range(Ne):
        id = icon[:, i]
        for k1 in range(npe[0]):
            for k2 in range(npe[1]):
                for k3 in range(npe[2]):
                    xi = np.array([k1, k2, k3]) * wt
                    n = shape3D(xi[0], xi[1], xi[2])
                    xtest = np.array([  np.dot(Xg[id-1, 0], n)
                                      , np.dot(Xg[id-1, 1], n),
                                        np.dot(Xg[id-1, 2], n)])
                    dis = np.linalg.norm(xtest - np.array([0.5, 0.5, 0.2]))
                    if dis <= 0.15:
                        xp.append(xtest)
                        vp.append(np.zeros_like(xtest))
    num_p = len(xp)
    xp = np.array(xp).T
    vp = np.array(vp).T
    return num_p, xp, vp