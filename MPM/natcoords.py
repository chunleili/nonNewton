import numpy as np


def natcoords(xp, dx, xmin, ne):
    n_p = xp.shape[1]
    nep = np.zeros((1, n_p))
    xpn = np.zeros((3, n_p))
    for ip in range(n_p):
        xx = (xp[:, ip] - xmin.T) / dx.T
        i = np.floor(xx)  # x y z element rough label
        nep[:, ip] = i[0] + i[1] * ne[0] + i[2] * (ne[1] * ne[0]) + 1  # element label
        id = nep[:, ip]
        xg = i.T * dx  # Xg(con(1,id),:); % left-front-bottom node
        xpn[:, ip] = 2 * (xp[:, ip] - xg.T) / dx.T - 1  # relative position to element center node
    return xpn, nep
