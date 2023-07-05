import numpy as np
import scipy


def massA2(dx, Ng, Ne, icon, F, We, dt):
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
