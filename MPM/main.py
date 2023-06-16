import numpy as np
from numpy import zeros, ones, kron, ones, arange
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend, quiver, quiverkey, gca, grid, xlim, ylim, subplot, axis, savefig
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from InitialParticle import InitialParticle
from FEMmatrix3D import FEMmatrix3D

# problem inputs
## ====time stepping====
dt=1e-4;#CFL * min(lx,ly)/1;
t=0;Tmax =100.0
Tmax=Tmax+dt
yita=1;  #yita=0: Newtonian yita=1: non-Newtonian
## ===========Flow parameters===============
Re=1e-2; beta=0.5;alph=0;gamma=0.9;Q0=20;We=0.01; 
xi=(1-beta)/(Re*We);
g=-9.8;Fr=0.1;
v0=-200;
## ========space discretization==================
xa=0; xb=1; # x direction 
ya=0; yb=1; # y direction
za=0; zb=1; # z direction
nx=10; ny=10; nz=10; # elements at x-y-z directions
Ne=nx*ny*nz; # total elemennts
Ng=(nx+1)*(ny+1)*(nz+1); # total nodes
Lx=(xb-xa)/nx; # dx 
Ly=(yb-ya)/ny; # dy
Lz=(zb-za)/nz; # dz 
## ======Ordering: (1) x y loop (2) z loop 
xe = np.arange(xa,xb+Lx,Lx).T # xe: x position sampling
ye = np.arange(ya,yb+Ly,Ly).T # ye: y position sampling
ze = np.arange(za,zb+Lz,Lz).T # ze: z position sampling
Ix = np.ones(nx+1)
Iy = np.ones(ny+1)
Iz = np.ones(nz+1)
xg=np.kron(Iz,np.kron(Iy,xe)); # xg: all nodes x position
yg=np.kron(Iz,np.kron(ye,Ix));# yg: all nodes x position
zg=np.kron(ze,np.kron(Iy,Ix));# yg: all nodes x position
## ==================
xmin=[xa,ya,za]; dx=[Lx,Ly,Lz]; nexyz=[nx,ny,ny];
Xg=np.vstack((xg,yg,zg)).T #all nodes position
Vg=np.zeros((3*Ng,1)); # velocity on the grids- vector [Vx; Vy; Vz];
VBC=0*Vg; # BC value
## connectivity matrix
icon=np.zeros((8, Ne),dtype=int);
#====first four nodes label on x-y layer======
for k in range(0,nz):
    for j in range(0,ny):
        for i in range(0,nx):
            e=(k)*ny*nx + (j)*nx + i
            icon[0,e] = (k)*(nx+1)*(ny+1) + (j)*(nx+1) + i + 1
            icon[1,e] = icon[0,e]+1
            icon[2,e] = icon[1,e]+nx+1
            icon[3,e] = icon[2,e]-1
#=====adding one x-y face total nodes to get other four nodes
icon[4:8,:] = icon[0:4,:] + (nx+1)*(ny+1); #

# iniitalize the material point
nep=np.array([8,8,8]);  
[Np, xp, vp]= InitialParticle(nep, dx, icon, Xg, Ne)
Tp=zeros((Np,6))
pp=zeros((Np,1))
#%%===========================================
figure(num='Initial mesh grid and mp');
ax = plt.axes(projection='3d')
for j in range(Ne):
    id=icon[:,j];    id=np.append(id, id[0])
    # ax.scatter(Xg[id,0], Xg[id,1], Xg[id,2], c='r', s=1)
    plt.plot(Xg[id-1, 0], Xg[id-1, 1], Xg[id-1, 2], 'r.', linewidth=1)
# plt.show()

#%% === boundary setup======================
ntop = np.where(Xg[:, 2] >= 1)[0]
nbot = np.where(Xg[:, 2] <= 0)[0]
nright = np.where(Xg[:, 0] >= 1)[0]
nleft = np.where(Xg[:, 0] <= 0)[0]
nfront = np.where(Xg[:, 1] <= 0)[0]
nback = np.where(Xg[:, 1] >= 1)[0]

NBCv = nbot + 2 * Ng

# %% =============Assemble Matrix============
[M, K, Gx, Gy, Gz] = FEMmatrix3D(Ne, Ng, icon, Lx, Ly, Lz);   

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

zer = np.zeros_like(Gx)
G10 = csr_matrix(
    np.vstack([np.hstack([Gx, zer, zer]),
               np.hstack([zer, Gy, zer]),
               np.hstack([zer, zer, Gz]),
               np.hstack([0.5*Gy, 0.5*Gx, zer]),
               np.hstack([0.5*Gz, zer, 0.5*Gx]),
               np.hstack([zer, 0.5*Gz, 0.5*Gy])])
)

G20 = csr_matrix(np.vstack([Gx, Gy, Gz]))

G30 = csr_matrix(
    np.vstack([np.hstack([Gx, zer, zer, Gy, Gz, zer]),
               np.hstack([zer, Gy, zer, Gz, zer, Gx]),
               np.hstack([zer, zer, Gz, zer, Gx, Gy])])
)