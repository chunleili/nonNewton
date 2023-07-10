import numpy as np
import taichi as ti


def natcoords(xp, dx, xmin, ne):
    n_p = xp.shape[1]
    nep = np.zeros((1, n_p), int)
    xpn = np.zeros((3, n_p))
    for ip in range(n_p):
        xx = (xp[:, ip] - xmin.T) / dx.T
        i = np.fix(xx).astype(int)  # x y z element rough label
        nep[:, ip] = i[0] + i[1] * ne[0] + i[2] * (ne[1] * ne[0]) + 1  # element label
        id = nep[:, ip]
        xg = i.T * dx  # Xg(con(1,id),:); % left-front-bottom node
        xpn[:, ip] = 2 * (xp[:, ip] - xg.T) / dx.T - 1  # relative position to element center node
    return xpn, nep


@ti.func
def ti_fix(x):
    res = ti.floor(x, ti.i32)
    for i in ti.static(range(3)):
        xx = x[i]
        if xx > 0:
            res[i] = ti.floor(xx, ti.i32)
        else:
            res[i] = ti.ceil(xx, ti.i32)
    return res


@ti.func
def natcoords_func(xp: ti.math.vec3, dx: ti.math.vec3, xmin: ti.math.vec3, ne: ti.math.ivec3):
    """Caution: This is not original natcoords function, but a version for only one point(vec3).
    I use this because I found it was one point all the time when called in a loop.
    """
    nep = 0
    xpn = ti.math.vec3(0.0)
    xx = (xp - xmin) / dx
    i = ti_fix(xx)  # x y z element rough label
    nep = i[0] + i[1] * ne[0] + i[2] * (ne[1] * ne[0]) + 1  # element label
    id = nep
    xg = i * dx  # element wise multiplication
    xpn = 2 * (xp - xg) / dx - 1  # relative position to element center node
