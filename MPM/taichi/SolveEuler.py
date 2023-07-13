import numpy as np
import scipy


def SolveEuler(X10, X20, X30, B1, B2, B3, A11, A12, A13, A21, A22, A23, A31, A32, A33, beta, Re, NBS, nd, nk, yita):
    nnx = nd[:, 0]
    nny = nd[:, 1]
    nnz = nd[:, 2]
    X1 = X10
    X2 = X20
    X3 = X30.flatten()
    A = scipy.sparse.bmat([[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]])
    B = np.concatenate([B1, B2, B3], axis=0)
    n = 2
    # --------------- Use Gauss-Seidel iterative way to solve AX=B --------------- #
    # ---------------- n=1 and eta=0 provides GS4 explicit scheme ---------------- #
    for i in range(n):
        RHS1 = B1 - A13 @ X3 - yita * A12 @ X2
        # X1 = scipy.sparse.linalg.spsolve(A11, RHS1)
        X1 = scipy.linalg.solve(A11.todense(), RHS1)

        RHS2 = B2 - A21 @ X1 - A23 @ X3
        if yita == 1:
            # X2 = scipy.sparse.linalg.spsolve(A22, RHS2)
            X2 = scipy.linalg.solve(A22.todense(), RHS2)
        else:
            X2 = 0 * RHS2
        R3 = B3 - A31 @ X1 - yita * A32 @ X2
        if yita == 1:
            l1 = X1[NBS] * nnx[NBS] ** 2
            l2 = X1[NBS + 3 * nk] * nnx[NBS] * nny[NBS]
            l3 = X1[NBS + 4 * nk] * nnx[NBS] * nnz[NBS]
            l4 = X1[NBS + 3 * nk] * nnx[NBS] * nny[NBS]
            l5 = X1[NBS + nk] * nny[NBS] ** 2
            l6 = X1[NBS + 5 * nk] * nny[NBS] * nnz[NBS]
            l7 = X1[NBS + 4 * nk] * nnx[NBS] * nnz[NBS]
            l8 = X1[NBS + 5 * nk] * nny[NBS] * nnz[NBS]
            l9 = X1[NBS + 2 * nk] * nnz[NBS] ** 2
            part1 = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9
            part1 = part1 * 2 * beta / Re

            m1 = X2[NBS] * nnx[NBS] ** 2
            m2 = X2[NBS + 3 * nk] * nnx[NBS] * nny[NBS]
            m3 = X2[NBS + 4 * nk] * nnx[NBS] * nnz[NBS]
            m4 = X2[NBS + 3 * nk] * nnx[NBS] * nny[NBS]
            m5 = X2[NBS + nk] * nny[NBS] ** 2
            m6 = X2[NBS + 5 * nk] * nny[NBS] * nnz[NBS]
            m7 = X2[NBS + 4 * nk] * nnx[NBS] * nnz[NBS]
            m8 = X2[NBS + 5 * nk] * nny[NBS] * nnz[NBS]
            m9 = X2[NBS + 2 * nk] * nnz[NBS] ** 2
            part2 = m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8 + m9
            part2 = part2 * yita

            concat = part1 + part2
            R3[NBS] = concat
        else:
            l1 = X1[NBS] * nnx[NBS] ** 2
            l2 = X1[NBS + 3 * nk] * nnx[NBS] * nny[NBS]
            l3 = X1[NBS + 4 * nk] * nnx[NBS] * nnz[NBS]
            l4 = X1[NBS + 3 * nk] * nnx[NBS] * nny[NBS]
            l5 = X1[NBS + nk] * nny[NBS] ** 2
            l6 = X1[NBS + 5 * nk] * nny[NBS] * nnz[NBS]
            l7 = X1[NBS + 4 * nk] * nnx[NBS] * nnz[NBS]
            l8 = X1[NBS + 5 * nk] * nny[NBS] * nnz[NBS]
            l9 = X1[NBS + 2 * nk] * nnz[NBS] ** 2
            part1 = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9
            part1 = part1 * 2 * beta / Re
            R3[NBS] = part1
        # X3 = scipy.sparse.linalg.spsolve(A33, R3)
        X3 = scipy.linalg.solve(A33.todense(), R3)
        # ----------------------------- check convergence ---------------------------- #
        X = np.concatenate([X1, X2, X3], axis=0)
        B = np.concatenate([B1, B2, B3], axis=0)
        R = np.linalg.norm(A @ X - B)
        if R <= 1e-7:
            break
    X = np.concatenate([X1, X2, X3], axis=0)
    return X1, X2, X3
