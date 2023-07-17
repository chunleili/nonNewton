import numpy as np
from scipy import sparse


def FEMmatrix3D(Ne, Ng, icon, Lx, Ly, Lz):
    M = sparse.lil_matrix((Ng, Ng))
    K = sparse.lil_matrix((Ng, Ng))
    Gx = sparse.lil_matrix((Ng, Ng))
    Gy = sparse.lil_matrix((Ng, Ng))
    Gz = sparse.lil_matrix((Ng, Ng))

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

    Kxe = (
        np.array(
            [
                [4, -4, -2, 2, 2, -2, -1, 1],
                [-4, 4, 2, -2, -2, 2, 1, -1],
                [-2, 2, 4, -4, -1, 1, 2, -2],
                [2, -2, -4, 4, 1, -1, -2, 2],
                [2, -2, -1, 1, 4, -4, -2, 2],
                [-2, 2, 1, -1, -4, 4, 2, -2],
                [-1, 1, 2, -2, -2, 2, 4, -4],
                [1, -1, -2, 2, 2, -2, -4, 4],
            ]
        )
        * Ly
        * Lz
        / Lx
        / 2
        / 18
    )

    Kye = (
        np.array(
            [
                [4, 2, -2, -4, 2, 1, -1, -2],
                [2, 4, -4, -2, 1, 2, -2, -1],
                [-2, -4, 4, 2, -1, -2, 2, 1],
                [-4, -2, 2, 4, -2, -1, 1, 2],
                [2, 1, -1, -2, 4, 2, -2, -4],
                [1, 2, -2, -1, 2, 4, -4, -2],
                [-1, -2, 2, 1, -2, -4, 4, 2],
                [-2, -1, 1, 2, -4, -2, 2, 4],
            ]
        )
        * Lx
        * Lz
        / Ly
        / 2
        / 18
    )

    Kze = (
        np.array(
            [
                [4, 2, 1, 2, -4, -2, -1, -2],
                [2, 4, 2, 1, -2, -4, -2, -1],
                [1, 2, 4, 2, -1, -2, -4, -2],
                [2, 1, 2, 4, -2, -1, -2, -4],
                [-4, -2, -1, -2, 4, 2, 1, 2],
                [-2, -4, -2, -1, 2, 4, 2, 1],
                [-1, -2, -4, -2, 1, 2, 4, 2],
                [-2, -1, -2, -4, 2, 1, 2, 4],
            ]
        )
        * Lx
        * Ly
        / Lz
        / 2
        / 18
    )

    Ke = Kxe + Kye + Kze
    Gxe = (
        np.array(
            [
                [-4, 4, 2, -2, -2, 2, 1, -1],
                [-4, 4, 2, -2, -2, 2, 1, -1],
                [-2, 2, 4, -4, -1, 1, 2, -2],
                [-2, 2, 4, -4, -1, 1, 2, -2],
                [-2, 2, 1, -1, -4, 4, 2, -2],
                [-2, 2, 1, -1, -4, 4, 2, -2],
                [-1, 1, 2, -2, -2, 2, 4, -4],
                [-1, 1, 2, -2, -2, 2, 4, -4],
            ]
        )
        * Ly
        * Lz
        / 4
        / 18
    )

    Gye = (
        np.array(
            [
                [-4, -2, 2, 4, -2, -1, 1, 2],
                [-2, -4, 4, 2, -1, -2, 2, 1],
                [-2, -4, 4, 2, -1, -2, 2, 1],
                [-4, -2, 2, 4, -2, -1, 1, 2],
                [-2, -1, 1, 2, -4, -2, 2, 4],
                [-1, -2, 2, 1, -2, -4, 4, 2],
                [-1, -2, 2, 1, -2, -4, 4, 2],
                [-2, -1, 1, 2, -4, -2, 2, 4],
            ]
        )
        * Lx
        * Lz
        / 4
        / 18
    )

    Gze = (
        np.array(
            [
                [-4, -2, -1, -2, 4, 2, 1, 2],
                [-2, -4, -2, -1, 2, 4, 2, 1],
                [-1, -2, -4, -2, 1, 2, 4, 2],
                [-2, -1, -2, -4, 2, 1, 2, 4],
                [-4, -2, -1, -2, 4, 2, 1, 2],
                [-2, -4, -2, -1, 2, 4, 2, 1],
                [-1, -2, -4, -2, 1, 2, 4, 2],
                [-2, -1, -2, -4, 2, 1, 2, 4],
            ]
        )
        * Lx
        * Ly
        / 4
        / 18
    )

    for h in range(Ne):
        id = icon[h, :] - 1
        for i in range(len(id)):
            for j in range(len(id)):
                if id[i] < Ng and id[j] < Ng:
                    M[id[i], id[j]] += Me[i, j]
                    K[id[i], id[j]] += Ke[i, j]
                    Gx[id[i], id[j]] += Gxe[i, j]
                    Gy[id[i], id[j]] += Gye[i, j]
                    Gz[id[i], id[j]] += Gze[i, j]

    M = sparse.csr_matrix(M)
    K = sparse.csr_matrix(K)
    Gx = sparse.csr_matrix(Gx)
    Gy = sparse.csr_matrix(Gy)
    Gz = sparse.csr_matrix(Gz)

    return M, K, Gx, Gy, Gz
