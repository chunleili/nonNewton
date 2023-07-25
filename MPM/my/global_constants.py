import numpy as np

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
Lx = (xb - xa) / nx  # dx
Ly = (yb - ya) / ny  # dy
Lz = (zb - za) / nz  # dz

dx = np.array([Lx, Ly, Lz])
xmin = np.array([xa, ya, za])
nexyz = np.array([nx, ny, ny])
