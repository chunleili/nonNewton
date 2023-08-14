import numpy as np
import taichi as ti

from shape3D import shape3D, shape3D_func


def InitialParticle_np(npe, dx, icon, Xg, Ne):
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
                    xtest = np.array([np.dot(Xg[id - 1, 0], n), np.dot(Xg[id - 1, 1], n), np.dot(Xg[id - 1, 2], n)])
                    dis = np.linalg.norm(xtest - np.array([0.5, 0.5, 0.2]))
                    if dis <= 0.15:
                        xp.append(xtest)
                        vp.append(np.zeros_like(xtest))
    num_p = len(xp)
    xp = np.array(xp).T
    vp = np.array(vp).T
    return num_p, xp, vp


def InitialParticle(npe, dx, icon, Xg, Ne):
    # num_p, xp, vp = InitialParticle_ti(npe, dx, icon, Xg, Ne)

    import meshio

    m = meshio.read("./MPM/data/cow.ply")
    xp = m.points
    # swap y and z
    xp[:, 1], xp[:, 2] = xp[:, 2], xp[:, 1].copy()
    num_p = len(xp)
    vp = np.zeros_like(xp)
    return num_p, xp, vp


def InitialParticle_ti(npe, dx, icon, Xg, Ne):
    wt = 2.0 / npe[:]
    xp = ti.ndarray(shape=(Ne * npe[0] * npe[1] * npe[2]), dtype=ti.math.vec3)
    vp = ti.ndarray(shape=(Ne * npe[0] * npe[1] * npe[2]), dtype=ti.math.vec3)
    wt_ = ti.Vector([wt[0], wt[1], wt[2]])
    num_p = InitialParticle_kernel(npe, dx, icon, Xg, Ne, wt_, xp, vp)
    xp = xp.to_numpy()
    xp = xp[:num_p]
    vp = vp.to_numpy()
    vp = vp[:num_p]
    return num_p, xp, vp


@ti.kernel
def InitialParticle_kernel(
    npe: ti.types.ndarray(),
    dx: ti.types.ndarray(),
    icon: ti.types.ndarray(),
    Xg: ti.types.ndarray(),
    Ne: int,
    wt: ti.types.vector(n=3, dtype=float),
    xp: ti.types.ndarray(),
    vp: ti.types.ndarray(),
) -> ti.i32:
    num_p = 0
    for i in range(Ne):
        id = get_col(icon, i)
        for k1 in range(npe[0]):
            for k2 in range(npe[1]):
                for k3 in range(npe[2]):
                    xi = ti.Vector([k1 * wt[0], k2 * wt[1], k3 * wt[2]])
                    n = shape3D_func(xi[0], xi[1], xi[2])
                    Xg0 = get_col_Xg_x(Xg, id)
                    Xg1 = get_col_Xg_y(Xg, id)
                    Xg2 = get_col_Xg_z(Xg, id)
                    Xg0_n = Xg0.dot(n)
                    Xg1_n = Xg1.dot(n)
                    Xg2_n = Xg2.dot(n)
                    xtest = ti.Vector([Xg0_n, Xg1_n, Xg2_n])
                    dis = (xtest - ti.Vector([0.5, 0.5, 0.2])).norm()
                    if dis <= 0.15:
                        xp[num_p] = xtest
                        vp[num_p] = ti.Vector([0, 0, 0])
                        num_p += 1
    return num_p


@ti.func
def get_col(icon: ti.template(), i: int):
    return ti.Vector([icon[i, 0], icon[i, 1], icon[i, 2], icon[i, 3], icon[i, 4], icon[i, 5], icon[i, 6], icon[i, 7]])


@ti.func
def get_col_Xg_x(Xg: ti.template(), id: ti.types.vector(8, dtype=float)):
    return ti.Vector(
        [
            Xg[id[0] - 1, 0],
            Xg[id[1] - 1, 0],
            Xg[id[2] - 1, 0],
            Xg[id[3] - 1, 0],
            Xg[id[4] - 1, 0],
            Xg[id[5] - 1, 0],
            Xg[id[6] - 1, 0],
            Xg[id[7] - 1, 0],
        ]
    )


@ti.func
def get_col_Xg_y(Xg: ti.template(), id: ti.types.vector(8, dtype=float)):
    return ti.Vector(
        [
            Xg[id[0] - 1, 1],
            Xg[id[1] - 1, 1],
            Xg[id[2] - 1, 1],
            Xg[id[3] - 1, 1],
            Xg[id[4] - 1, 1],
            Xg[id[5] - 1, 1],
            Xg[id[6] - 1, 1],
            Xg[id[7] - 1, 1],
        ]
    )


@ti.func
def get_col_Xg_z(Xg: ti.template(), id: ti.types.vector(8, dtype=float)):
    return ti.Vector(
        [
            Xg[id[0] - 1, 2],
            Xg[id[1] - 1, 2],
            Xg[id[2] - 1, 2],
            Xg[id[3] - 1, 2],
            Xg[id[4] - 1, 2],
            Xg[id[5] - 1, 2],
            Xg[id[6] - 1, 2],
            Xg[id[7] - 1, 2],
        ]
    )
