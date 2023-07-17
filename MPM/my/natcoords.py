import numpy as np
import taichi as ti
from time import time


def natcoords_np(xp, dx, xmin, ne):
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


def natcoords_ti(xp, dx, xmin, ne):
    # new var
    n_p = xp.shape[0]
    nep_ti = ti.ndarray(dtype=ti.i32, shape=(n_p))
    xpn_ti = ti.ndarray(dtype=ti.math.vec3, shape=(n_p))

    # old var
    xp_ti = ti.ndarray(dtype=ti.math.vec3, shape=(n_p))
    xp_ti.from_numpy(xp)
    dx_ti = ti.math.vec3(dx)
    xmin_ti = ti.math.vec3(xmin)
    ne_ti = ti.math.ivec3(ne)

    natcoords_kernel(xp_ti, dx_ti, xmin_ti, ne_ti, nep_ti, xpn_ti, n_p)

    # copy back
    xpn = xpn_ti.to_numpy().reshape(3, n_p)
    nep = nep_ti.to_numpy().reshape(1, n_p)
    return xpn, nep


@ti.kernel
def natcoords_kernel(
    xp: ti.types.ndarray(),
    dx: ti.math.vec3,
    xmin: ti.math.vec3,
    ne: ti.math.ivec3,
    nep: ti.types.ndarray(),
    xpn: ti.types.ndarray(),
    n_p: ti.i32,
):
    for ip in range(n_p):
        xx = (xp[ip] - xmin) / dx
        i = ti_fix(xx)
        nep[ip] = i[0] + i[1] * ne[0] + i[2] * (ne[1] * ne[0]) + 1
        xg = i * dx
        xpn[ip] = 2 * (xp[ip] - xg) / dx - 1


def natcoords(xp, dx, xmin, ne):
    # t = time()
    xpn, nep = natcoords_ti(xp, dx, xmin, ne)
    # print("natcoords_ti: ", time() - t)
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
    return xpn, nep
