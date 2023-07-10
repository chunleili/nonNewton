import numpy as np
import scipy
from time import time, perf_counter
import taichi as ti

from natcoords import natcoords, natcoords_func
from shape3D import shape3D, shape3D_func


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
    start_t = time()
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
    end_t = time()
    print("RK4: ", end_t - start_t)
    return xp, vp, Tp, pp


def maptopointsPC(Ng, Tg, xmin, nexyz, Np, xp, vp, icon, vnew, v, dx, dt, p, Fr, g):
    # get natural coordinates of particles within cell
    [xpn, nep] = natcoords(xp, dx, xmin, nexyz)
    nep = nep.flatten()

    vnew = np.vstack([vnew[0:Ng], vnew[Ng : 2 * Ng], vnew[2 * Ng : 3 * Ng]]).T
    v = np.vstack([v[0:Ng], v[Ng : 2 * Ng], v[2 * Ng : 3 * Ng]]).T
    nexyz = np.array(nexyz)

    # 处理输入值
    ivec8 = ti.types.vector(8, int)
    vec6 = ti.types.vector(6, float)
    vnew_ti = ti.ndarray(dtype=ti.math.vec3, shape=(Ng))
    vnew_ti.from_numpy(vnew)
    v_ti = ti.ndarray(dtype=ti.math.vec3, shape=(Ng))
    v_ti.from_numpy(v)
    icon_ti = ti.ndarray(dtype=ivec8, shape=(icon.shape[1]))
    icon_ti.from_numpy(icon.T)
    xp_ti = ti.ndarray(dtype=ti.math.vec3, shape=(xp.shape[1]))
    xp_ti.from_numpy(xp.T)
    dx_ti = ti.math.vec3([dx[0], dx[1], dx[2]])

    # 新建变量
    pp = ti.ndarray(dtype=ti.f32, shape=(Np))
    Tp = ti.ndarray(dtype=ti.f32, shape=(Np, 6))
    xpF = ti.ndarray(dtype=ti.math.vec3, shape=(Np))
    k1 = ti.ndarray(dtype=ti.math.vec3, shape=(Np))
    k2 = ti.ndarray(dtype=ti.math.vec3, shape=(Np))
    k3 = ti.ndarray(dtype=ti.math.vec3, shape=(Np))
    k4 = ti.ndarray(dtype=ti.math.vec3, shape=(Np))

    ## RK4
    start_t = time()
    maptopointsPC_kernel(
        Np,
        icon_ti,
        xp_ti,
        dt,
        dx_ti,
        xmin,
        nexyz,
        vp,
        p,
        g,
        Tg,
        Fr,
        Ng,
        xpn,
        nep,
        vnew_ti,
        v_ti,
        pp,
        Tp,
        xpF,
        k1,
        k2,
        k3,
        k4,
    )
    k1 = k1.to_numpy().T
    xp = xp_ti.to_numpy().T
    xpF = xpF.to_numpy().T
    # vp = vp_ti.to_numpy().T
    # pp = pp_ti.to_numpy().T
    end_t = time()
    print("RK4: ", end_t - start_t)
    return xp, vp, Tp, pp


@ti.func
def get_col(A: ti.types.ndarray(), j: ti.i32, ret: ti.types.ndarray()):
    for i in range(A.shape[0]):
        ret[i] = A[i, j]


@ti.func
def get_row(A: ti.types.ndarray(), i: ti.i32, ret: ti.types.ndarray()):
    for j in range(A.shape[1]):
        ret[j] = A[i, j]


@ti.func
def get_slice_8(arr: ti.types.ndarray(), index: ti.template(), axis: ti.i32):
    """
    arr: to be sliced
    index: index of the 8 points
    axis: x or y or z axis
    """
    return ti.Vector(
        [
            arr[index[0]][axis],
            arr[index[1]][axis],
            arr[index[2]][axis],
            arr[index[3]][axis],
            arr[index[4]][axis],
            arr[index[5]][axis],
            arr[index[6]][axis],
            arr[index[7]][axis],
        ]
    )


@ti.kernel
def maptopointsPC_kernel(
    Np: int,
    icon: ti.types.ndarray(),
    xp: ti.types.ndarray(),
    dt: float,
    dx: ti.types.vector(3, float),
    xmin: ti.types.vector(3, float),
    nexyz: ti.types.vector(3, int),
    vp: ti.types.ndarray(),
    p: ti.types.ndarray(),
    g: float,
    Tg: ti.types.ndarray(),
    Fr: float,
    Ng: int,
    xpn: ti.types.ndarray(),
    nep: ti.types.ndarray(),
    vnew: ti.types.ndarray(),
    v: ti.types.ndarray(),
    pp: ti.types.ndarray(),
    Tp: ti.types.ndarray(),
    xpF: ti.types.ndarray(),
    k1: ti.types.ndarray(),
    k2: ti.types.ndarray(),
    k3: ti.types.ndarray(),
    k4: ti.types.ndarray(),
):
    for ip in range(1):
        i = nep[ip]
        iv = icon[i]
        n = shape3D_func(xpn[ip, 0], xpn[ip, 1], xpn[ip, 2])
        v_iv_0 = get_slice_8(v, iv, 0)
        v_iv_1 = get_slice_8(v, iv, 1)
        v_iv_2 = get_slice_8(v, iv, 2)
        k1[ip][0] = v_iv_0.dot(n)
        k1[ip][1] = v_iv_1.dot(n)
        k1[ip][2] = v_iv_2.dot(n)
        xpF[ip] = xp[ip] + k1[ip] * dt / 2

        natcoords_func(xpF[ip], dx, xmin, nexyz)
