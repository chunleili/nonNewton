import numpy as np
import taichi as ti

from shape3D import shape3D, shape3D_func

mat8x6 = ti.types.matrix(8, 6, dtype=ti.f32)
mat8x3 = ti.types.matrix(8, 3, dtype=ti.f32)


def P2G_(Ng, icon, xpn, nep, n_p, vp, pp, Tp):
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


def P2G(Ng, icon, xpn, nep, n_p, vp, pp, Tp):
    v = np.zeros((Ng, 3))

    #  map velocities to grid
    nep = nep.flatten()

    # taichi variables
    ivec8 = ti.types.vector(8, int)
    vec6 = ti.types.vector(6, float)
    mv_ti = ti.ndarray(dtype=ti.f32, shape=(Ng))
    pv_ti = ti.ndarray(dtype=ti.math.vec3, shape=(Ng))
    Tv_ti = ti.ndarray(dtype=vec6, shape=(Ng))
    nn_ti = ti.ndarray(dtype=int, shape=(Ng))
    p_ti = ti.ndarray(dtype=ti.f32, shape=(Ng))

    icon_ti = ti.ndarray(dtype=ivec8, shape=(icon.shape[1]))
    icon_ti.from_numpy(icon.T)
    xpn_ti = ti.ndarray(dtype=ti.math.vec3, shape=(xpn.shape[1]))
    xpn_ti.from_numpy(xpn.T)
    vp_ti = ti.ndarray(dtype=ti.math.vec3, shape=(vp.shape[1]))
    vp_ti.from_numpy(vp.T)
    pp_ti = ti.ndarray(dtype=ti.f32, shape=(pp.shape[0]))
    pp_ti.from_numpy(pp.flatten())
    Tp_ti = ti.ndarray(dtype=vec6, shape=(Tp.shape[0]))
    Tp_ti.from_numpy(Tp)

    P2G_kernel(n_p, nep, icon_ti, xpn_ti, mv_ti, pv_ti, Tv_ti, vp_ti, nn_ti, pp_ti, p_ti, Tp_ti)

    # copy back to numpy
    mv = mv_ti.to_numpy().reshape(-1, 1)
    pv = pv_ti.to_numpy()
    Tv = Tv_ti.to_numpy()
    nn = nn_ti.to_numpy().reshape(-1, 1)
    p = p_ti.to_numpy().reshape(-1, 1)

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


@ti.func
def get_slice_8_scalar(arr: ti.types.ndarray(), index: ti.template()):
    """
    arr: to be sliced
    index: index of the 8 points
    """
    return ti.Vector(
        [
            arr[index[0]],
            arr[index[1]],
            arr[index[2]],
            arr[index[3]],
            arr[index[4]],
            arr[index[5]],
            arr[index[6]],
            arr[index[7]],
        ]
    )


@ti.func
def set_slice_8(arr: ti.types.ndarray(), index: ti.template(), val: ti.template()):
    """
    arr: to be sliced
    index: index of the 8 points
    """
    for dim in ti.static(range(8)):
        arr[index[dim]] = val


@ti.func
def atomic_add_slice_8_scalar(arr: ti.types.ndarray(), index: ti.template(), val: ti.template()):
    """
    arr: to be sliced
    index: index of the 8 points
    """
    arr[index[0]] += val[0]
    arr[index[1]] += val[1]
    arr[index[2]] += val[2]
    arr[index[3]] += val[3]
    arr[index[4]] += val[4]
    arr[index[5]] += val[5]
    arr[index[6]] += val[6]
    arr[index[7]] += val[7]


@ti.func
def atomic_add_slice_8_mat(arr: ti.types.ndarray(), index: ti.template(), val: ti.template()):
    """
    arr: to be sliced
    index: index of the 8 points
    """
    arr[index[0]] += val[0, :]
    arr[index[1]] += val[1, :]
    arr[index[2]] += val[2, :]
    arr[index[3]] += val[3, :]
    arr[index[4]] += val[4, :]
    arr[index[5]] += val[5, :]
    arr[index[6]] += val[6, :]
    arr[index[7]] += val[7, :]


@ti.kernel
def P2G_kernel(
    n_p: ti.i32,
    nep: ti.types.ndarray(),
    icon: ti.types.ndarray(),
    xpn: ti.types.ndarray(),
    mv: ti.types.ndarray(),
    pv: ti.types.ndarray(),
    Tv: ti.types.ndarray(),
    vp: ti.types.ndarray(),
    nn: ti.types.ndarray(),
    pp: ti.types.ndarray(),
    p: ti.types.ndarray(),
    Tp: ti.types.ndarray(),
):
    for ip in range(n_p):
        i = nep[ip]
        iv = icon[i]
        n = shape3D_func(xpn[ip][0], xpn[ip][1], xpn[ip][2])

        atomic_add_slice_8_scalar(mv, iv, n)

        pv_mat = mat8x3(0.0)
        for dim in ti.static(range(3)):
            pv_mat[:, dim] = vp[ip][dim] * n  # each line is vec8
        atomic_add_slice_8_mat(pv, iv, pv_mat)

        Tp_mat = mat8x6(0.0)
        for dim in ti.static(range(6)):
            Tp_mat[:, dim] = Tp[ip][dim] * n  # each line is vec8
        atomic_add_slice_8_mat(Tv, iv, Tp_mat)

        for dim in ti.static(range(8)):
            nn[iv[dim]] = int(1)

        atomic_add_slice_8_scalar(p, iv, pp[ip] * n)


# def P2G(Ng, icon, xpn, nep, n_p, vp, pp, Tp):
#     mv_np, v_np, T1_np, p_np, nn_np = P2G_np(Ng, icon, xpn, nep, n_p, vp, pp, Tp)
#     mv, v, T1, p, nn = P2G_ti(Ng, icon, xpn, nep, n_p, vp, pp, Tp)
#     assert np.allclose(mv_np, mv)
#     assert np.allclose(v_np, v)
#     assert np.allclose(T1_np, T1)
#     assert np.allclose(p_np, p)
#     assert np.array_equal(nn_np, nn)
#     ...
