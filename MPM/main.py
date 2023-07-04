import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from time import time

from InitialParticle import InitialParticle
from FEMmatrix3D import FEMmatrix3D
from natcoords import natcoords
from P2G import P2G
from Hstar import Hstar
from massA2 import massA2
from full_scale_operators import full_scale_operators
from Matrix_reduction import Matrix_reduction
from SolveAssemble import SolveAssemble
from SolveEuler import SolveEuler
from maptopointsPC import maptopointsPC


def main():
    # problem inputs
    ## ====time stepping====
    dt = 1e-4
    # CFL * min(lx,ly)/1;
    t = 0
    Tmax = 100.0
    Tmax = Tmax + dt
    yita = 1
    # yita=0: Newtonian yita=1: non-Newtonian
    ## ===========Flow parameters===============
    Re = 1e-2
    beta = 0.5
    alph = 0
    gamma = 0.9
    Q0 = 20
    We = 0.01
    xi = (1 - beta) / (Re * We)
    g = -9.8
    Fr = 0.1
    v0 = -200
    ## ========space discretization==================
    xa = 0
    xb = 1
    # x direction
    ya = 0
    yb = 1
    # y direction
    za = 0
    zb = 1
    # z direction
    nx = 10
    ny = 10
    nz = 10
    # elements at x-y-z directions
    Ne = nx * ny * nz
    # total elemennts
    Ng = (nx + 1) * (ny + 1) * (nz + 1)
    # total nodes
    Lx = (xb - xa) / nx
    # dx
    Ly = (yb - ya) / ny
    # dy
    Lz = (zb - za) / nz
    # dz
    ## ======Ordering: (1) x y loop (2) z loop
    xe = np.arange(xa, xb + Lx, Lx).T  # xe: x position sampling
    ye = np.arange(ya, yb + Ly, Ly).T  # ye: y position sampling
    ze = np.arange(za, zb + Lz, Lz).T  # ze: z position sampling
    Ix = np.ones(nx + 1)
    Iy = np.ones(ny + 1)
    Iz = np.ones(nz + 1)
    xg = np.kron(Iz, np.kron(Iy, xe))
    # xg: all nodes x position
    yg = np.kron(Iz, np.kron(ye, Ix))
    # yg: all nodes x position
    zg = np.kron(ze, np.kron(Iy, Ix))
    # yg: all nodes x position
    ## ==================
    xmin = np.array([xa, ya, za])
    dx = np.array([Lx, Ly, Lz])
    nexyz = [nx, ny, ny]
    Xg = np.vstack((xg, yg, zg)).T  # all nodes position
    Vg = np.zeros((3 * Ng, 1))
    # velocity on the grids- vector [Vx; Vy; Vz];
    VBC = 0 * Vg
    # BC value
    ## connectivity matrix
    icon = np.zeros((8, Ne), dtype=int)
    # ====first four nodes label on x-y layer======
    for k in range(0, nz):
        for j in range(0, ny):
            for i in range(0, nx):
                e = (k) * ny * nx + (j) * nx + i
                icon[0, e] = (k) * (nx + 1) * (ny + 1) + (j) * (nx + 1) + i + 1
                icon[1, e] = icon[0, e] + 1
                icon[2, e] = icon[1, e] + nx + 1
                icon[3, e] = icon[2, e] - 1
    # =====adding one x-y face total nodes to get other four nodes
    icon[4:8, :] = icon[0:4, :] + (nx + 1) * (ny + 1)

    # iniitalize the material point
    nep = np.array([8, 8, 8])
    [Np, xp, vp] = InitialParticle(nep, dx, icon, Xg, Ne)
    Tp = np.zeros((Np, 6))
    pp = np.zeros((Np, 1))
    ## =========plot initial================
    plot_init(Xg, icon, Ne, True)

    ## === boundary setup======================
    ntop = np.where(Xg[:, 2] >= 1)[0]
    nbot = np.where(Xg[:, 2] <= 0)[0]
    nright = np.where(Xg[:, 0] >= 1)[0]
    nleft = np.where(Xg[:, 0] <= 0)[0]
    nfront = np.where(Xg[:, 1] <= 0)[0]
    nback = np.where(Xg[:, 1] >= 1)[0]

    NBCv = nbot + 2 * Ng

    ## =============Assemble Matrix============
    [M, K, Gx, Gy, Gz] = FEMmatrix3D(Ne, Ng, icon, Lx, Ly, Lz)

    zer = np.zeros(shape=Gx.shape)

    # Gradient of Velocity
    l1 = scipy.sparse.hstack([Gx, zer, zer])
    l2 = scipy.sparse.hstack([zer, Gy, zer])
    l3 = scipy.sparse.hstack([zer, zer, Gz])
    l4 = scipy.sparse.hstack([0.5 * Gy, 0.5 * Gx, zer])
    l5 = scipy.sparse.hstack([0.5 * Gz, zer, 0.5 * Gx])
    l6 = scipy.sparse.hstack([zer, 0.5 * Gz, 0.5 * Gy])
    G10 = scipy.sparse.vstack([l1, l2, l3, l4, l5, l6])
    # Gradient of P
    G20 = scipy.sparse.vstack([Gx, Gy, Gz])
    # Divergence of stress
    l1 = scipy.sparse.hstack([Gx, zer, zer, Gy, Gz, zer])
    l2 = scipy.sparse.hstack([zer, Gy, zer, Gz, zer, Gx])
    l3 = scipy.sparse.hstack([zer, zer, Gz, zer, Gx, Gy])
    G30 = scipy.sparse.vstack([l1, l2, l3])

    # mass matrix
    Mu0 = scipy.sparse.block_diag([M, M, M])
    Mt0 = scipy.sparse.block_diag([M, M, M, M, M, M])
    Mu = np.sum(Mu0, axis=1)
    Mu = np.squeeze(np.asarray(Mu))
    inv_Mu = 1.0 / Mu[:]
    dMu = scipy.sparse.diags(inv_Mu)
    Mt0 = scipy.sparse.diags(np.squeeze(np.asarray(np.sum(Mt0, axis=1))))

    p = np.zeros((Np, 1))
    vp[2, :] = v0
    step = 0
    starttime = time()
    while step < 200:
        step += 1
        print("step: ", step)
        # Map to grid
        [xpn, nep] = natcoords(xp, dx, xmin, nexyz)
        [mv, vnew, Tnew, p, nn] = P2G(Ng, icon, xpn, nep, Np, vp, pp, Tp)
        for i in range(len(NBCv)):
            if vnew[NBCv[i]] < 0:
                vnew[NBCv[i]] = VBC[NBCv[i]]
        v = vnew
        T = Tnew
        [Hg, f] = Hstar(T, alph, gamma, Q0, xi, We, Ng, G10, vnew, Mt0)
        [Mf, A22d] = massA2(dx, Ng, Ne, icon, f, We, dt)
        NBC = np.where(nn <= 0)[0]  # NBC: non particle nodes, NBS: free surface nodes
        Phi = scipy.sparse.eye(Ng)
        keep_columns = np.ones(Phi.shape[1], dtype=bool)
        keep_columns[NBC] = False
        Phi_subset = Phi.tocsr()
        Phi_subset = Phi_subset[:, keep_columns]
        Phi = Phi_subset
        dxm = Phi.T @ Gx @ Phi
        dym = Phi.T @ Gy @ Phi
        dzm = Phi.T @ Gz @ Phi
        nk = dxm.shape[0]
        Ck = Phi.T @ M @ Phi
        zk = np.zeros((nk, nk))
        [G1, D, G, Mu, Md, Mt, dMu, A22d] = full_scale_operators(dxm, dym, dzm, Ck, Mf, Phi, Re, beta, zk)
        [Nk, vk, Hk, Tk, pk, znk] = Matrix_reduction(nn, nk, Ng, Phi, vnew, Hg, Tnew, p)

        ## ==========Free Surface================
        Gnx = dxm @ Nk
        Gny = dym @ Nk
        Gnz = dzm @ Nk
        Gn = np.sqrt(Gnx * Gnx + Gny * Gny + Gnz * Gnz)
        nnx = Gnx / (Gn)
        nny = Gny / (Gn)
        nnz = Gnz / (Gn)
        NBS = np.where(Gn > 1e-8)[0]
        nd = np.hstack([nnx, nny, nnz])
        X10 = beta / Re * G1 @ vk
        X20 = Tk
        X30 = pk.flatten()

        ## ===========Solve ============
        [A11, A12, A13, A21, A22, A23, A31, A32, A33, B1, B2, B3] = SolveAssemble(
            X20, Hk, dMu, Mt, A22d, NBS, beta, Re, G1, D, G, xi, nk, vk, dt
        )
        [X1, X2, X3] = SolveEuler(
            X10, X20, X30, B1, B2, B3, A11, A12, A13, A21, A22, A23, A31, A32, A33, beta, Re, NBS, nd, nk, yita
        )
        ## ==========Results Reterive====
        zn = np.zeros((Ng, nk))
        Tg = scipy.sparse.block_diag([Phi, Phi, Phi, Phi, Phi, Phi]) @ X2
        p = Phi @ X3
        dV = dMu @ (beta / Re * D @ X1 + D @ X2 - G @ X3) * dt
        deltV = scipy.sparse.block_diag([Phi, Phi, Phi]) @ dV
        vtilda = vnew + deltV
        vnew = vtilda

        ## ==========Boundary Condition==========
        for i in range(len(NBCv)):
            if vnew[NBCv[i]] < 0:
                vnew[NBCv[i]] = VBC[NBCv[i]]
        ## ===== Grids to Particles=============
        [xp, vp, Tp, pp] = maptopointsPC(Ng, Tg, xmin, nexyz, Np, xp, vp, icon, vnew, v, dx, dt, p, Fr, g)
        v = vnew
        ## ====Plot======
        plot_step(xp)
    endtime = time()
    print("total time used: ", endtime - starttime)


def plot_step(xp, plot=True):
    if not plot:
        return
    plt.ion()
    ax = plt.axes(projection="3d")
    ax.scatter(xp[0, :], xp[1, :], xp[2, :], c="b")
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.2, 0.8)
    ax.set_zlim(0, 0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([2, 2, 1])
    plt.show()
    plt.pause(1)
    plt.clf()


def plot_init(Xg, icon, Ne, plot=True):
    if not plot:
        return
    plt.figure(num="Initial mesh grid and mp")
    ax = plt.axes(projection="3d")
    for j in range(Ne):
        id = icon[:, j]
        id = np.append(id, id[0])
        ax.scatter(Xg[id - 1, 0], Xg[id - 1, 1], Xg[id - 1, 2], c="r", s=1)
    plt.show()


if __name__ == "__main__":
    main()
