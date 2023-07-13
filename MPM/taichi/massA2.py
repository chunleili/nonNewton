import numpy as np
import scipy
import taichi as ti
from time import time


def massA2_np(dx, Ng, Ne, icon, F, We, dt):
    Lx = dx[0]
    Ly = dx[1]
    Lz = dx[2]
    C = np.zeros((Ng, Ng))
    # Npt*Np on all elements
    Me = (
        np.array(
            [
                [8, 4, 2, 4, 4, 2, 1, 2],
                [4, 8, 4, 2, 2, 4, 2, 1],
                [2, 4, 8, 4, 1, 2, 4, 2],
                [4, 2, 4, 8, 2, 1, 2, 4],
                [4, 2, 1, 2, 8, 4, 2, 4],
                [2, 4, 2, 1, 4, 8, 4, 2],
                [1, 2, 4, 2, 2, 4, 8, 4],
                [2, 1, 2, 4, 4, 2, 4, 8],
            ]
        )
        * Lx
        * Ly
        * Lz
        / 8
        / 27
    )
    for i in range(Ne):
        nodes = icon[:, i] - 1
        f = dt * np.sum(F[nodes]) / 8 / We

        C[nodes[:, None], nodes] = C[nodes[:, None], nodes] + f * Me

    C = scipy.sparse.bsr_matrix(C)
    z = np.zeros((Ng, Ng))
    l1 = scipy.sparse.hstack([C, z, z, z, z, z])
    l2 = scipy.sparse.hstack([z, C, z, z, z, z])
    l3 = scipy.sparse.hstack([z, z, C, z, z, z])
    l4 = scipy.sparse.hstack([z, z, z, C, z, z])
    l5 = scipy.sparse.hstack([z, z, z, z, z, C])
    M = scipy.sparse.vstack([l1, l2, l3, l4, l5])

    return C, M


def massA2_ti(dx, Ng, Ne, icon, F, We, dt):
    Lx = dx[0]
    Ly = dx[1]
    Lz = dx[2]
    C = np.zeros((Ng, Ng))
    # Npt*Np on all elements
    Me = (
        np.array(
            [
                [8, 4, 2, 4, 4, 2, 1, 2],
                [4, 8, 4, 2, 2, 4, 2, 1],
                [2, 4, 8, 4, 1, 2, 4, 2],
                [4, 2, 4, 8, 2, 1, 2, 4],
                [4, 2, 1, 2, 8, 4, 2, 4],
                [2, 4, 2, 1, 4, 8, 4, 2],
                [1, 2, 4, 2, 2, 4, 8, 4],
                [2, 1, 2, 4, 4, 2, 4, 8],
            ]
        )
        * Lx
        * Ly
        * Lz
        / 8
        / 27
    )

    ivec8 = ti.types.vector(8, ti.i32)
    icon_ti = ti.ndarray(dtype=ivec8, shape=(Ne))
    icon_ti.from_numpy(icon.T)
    massA2_kernel(icon_ti, F.flatten(), We, dt, Ne, Me, C)

    C = scipy.sparse.dia_matrix(C)
    # z = np.zeros((Ng, Ng))
    # l1 = scipy.sparse.hstack([C, z, z, z, z, z])
    # l2 = scipy.sparse.hstack([z, C, z, z, z, z])
    # l3 = scipy.sparse.hstack([z, z, C, z, z, z])
    # l4 = scipy.sparse.hstack([z, z, z, C, z, z])
    # l5 = scipy.sparse.hstack([z, z, z, z, z, C])
    # M = scipy.sparse.vstack([l1, l2, l3, l4, l5])
    return C


@ti.kernel
def massA2_kernel(
    icon: ti.types.ndarray(),
    F: ti.types.ndarray(),
    We: ti.f32,
    dt: ti.f32,
    Ne: ti.i32,
    Me: ti.types.ndarray(),
    C: ti.types.ndarray(),
):
    for i in range(Ne):
        nodes = icon[i] - 1

        f = (
            dt
            * (
                F[nodes[0]]
                + F[nodes[1]]
                + F[nodes[2]]
                + F[nodes[3]]
                + F[nodes[4]]
                + F[nodes[5]]
                + F[nodes[6]]
                + F[nodes[7]]
            )
            / 8
            / We
        )

        for dim in ti.static(range(8)):
            C[nodes[dim], nodes[dim]] += f * Me[dim, dim]


def massA2(dx, Ng, Ne, icon, F, We, dt):
    return massA2_ti(dx, Ng, Ne, icon, F, We, dt)


# def massA2(dx, Ng, Ne, icon, F, We, dt):
#     C_np, M_np = massA2_np(dx, Ng, Ne, icon, F, We, dt)
#     C_ti, M_ti = massA2_ti(dx, Ng, Ne, icon, F, We, dt)
#     assert np.allclose(C_np.toarray(), C_ti.toarray(), atol=1e-5), np.where(C_np.toarray() != C_ti.toarray())
#     assert np.allclose(M_np.toarray(), M_ti.toarray(), atol=1e-5), np.where(M_np.toarray() != M_ti.toarray())
