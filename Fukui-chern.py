"""
Created on Thu Sep 28 14:34:46 2023
@author: yucheng
"""

import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib as mpl

## Set Haldane model parameters
M, t1, t2, phi = 0.7, 0.2, -1, np.pi*0.5

## Set k-grid parameters
#  number of grids on reciprocal lattice vectors
num_b1 = 10
num_b2 = 10
# Reciprocal lattice vectors when real-space A,B sub-lattice distance=1 (lattice constant set to unity)
b1 = 4*np.pi/3 * np.array([3**0.5/2, -1/2])
b2 = np.array([0, 4*np.pi/3])
vec_1 = b1 / num_b1
vec_2 = b2 / num_b2


## build a function that takes in Hamiltonian and gives GS
def GS_vec(H):
    eigen_val, eigen_vec = LA.eig(H)
    #  sorted enumerate produce (original index, value) in sorted order
    sorted_val = sorted(enumerate(eigen_val), key=lambda i: i[1])
    return eigen_vec.T[sorted_val[0][0]]  #LA.eig returns eigenvectors in column

def pauli():
    # define puali matrices
    s_0=np.identity(2)
    s_z=np.array([[1,0],[0,-1]])
    s_x=np.array([[0,1],[1,0]])
    s_y=np.array([[0,-1j],[1j,0]])
    return s_0, s_x, s_y, s_z

## Haldane model in momentum space
#  lattice constant is set to unity, k=[kx, ky]
def Haldane(k, M, t1, t2, phi):
    # nearest neighbor displacement
    e1 = np.array([0,1])
    e2 = 0.5*np.array([-1*3**0.5, -1])
    e3 = 0.5*np.array([3**0.5, -1])
    # next nearest neighbor displacement
    v1 = np.array([-1*3**0.5, 0])
    v2 = 0.5*np.array([3**0.5, -3])
    v3 = 0.5*np.array([3**0.5, 3])
    f0 = 2*t2*math.cos(phi)*(np.cos(np.dot(k, v1)) + np.cos(np.dot(k, v2)) + np.cos(np.dot(k, v3)))
    f1 = t1*(np.cos(np.dot(k,e1)) + np.cos(np.dot(k,e2)) + np.cos(np.dot(k,e3)))
    f2 = t1*(np.sin(np.dot(k,e1)) + np.sin(np.dot(k,e2)) + np.sin(np.dot(k,e3)))
    f3 = M - 2*t2*np.sin(phi)*(np.sin(np.dot(k,v1)) + np.sin(np.dot(k,v2)) + np.sin(np.dot(k,v3)))
    s_0, s_x, s_y, s_z = pauli()
    return s_0*f0 + s_x*f1 + s_y*f2 + s_z*f3

## U(1) link variable Eq.(7) in [Fukui, 2005]
#  I write this first for 2x2 Hamiltonian -> only the lowwer band Chern number is needed
def U(mu, k_l):  # mu, k_l=[kx, ky]
    n_kl = GS_vec(Haldane(k_l, M, t1, t2, phi))
    n_klmu = GS_vec(Haldane(k_l + mu, M, t1, t2, phi))
    N_mukl = abs(np.dot(n_kl.conj(), n_klmu))
    return np.dot(n_kl.conj(), n_klmu) / N_mukl

def F_tilde_12(k_l):
    U_1kl = U(vec_1, k_l)
    U_2kl1 = U(vec_2, k_l + vec_1)
    U_1kl2 = U(vec_1, k_l + vec_2)
    U_2kl = U(vec_2, k_l)
    return np.log(U_1kl * U_2kl1 / (U_1kl2 * U_2kl))

def Chern():
    chern = 0
    for i in range(num_b1):
        for j in range(num_b2):
            chern += F_tilde_12(i*vec_1 + j*vec_2)
    return np.real(chern / (2*np.pi*1j))

print(Chern())

## Plot setting
M_list = np.linspace(-6, 6, num=30)
phi_list = np.linspace(-np.pi, np.pi, num=30)
grid_x, grid_y = np.meshgrid(phi_list, M_list)  #np.meshgrid(x_grid_1d, y_grid_1d) gives correct 2D grid for x,y
Chern_grid = np.zeros((len(phi_list), len(M_list)))  #np.zeros(y,x) gives y times x matrix
for i in range(len(phi_list)):
    print('phi_index= ', i)
    for j in range(len(M_list)):
        phi, M = phi_list[i], M_list[j]
        Chern_grid[j][i] = Chern()

cmap = plt.get_cmap('viridis')
zmin = -1
zmax = 1
norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)

fig, ax0 = plt.subplots(nrows=1)
im = ax0.pcolormesh(grid_x, grid_y, Chern_grid, cmap=cmap, norm=norm)
## Create colorbar
#  creating ScalarMappable
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ticks=np.linspace(zmin, zmax, 3))
## Title
ax0.set_title('Haldane model phase diagram')
plt.xlabel('phi', fontsize=12)
plt.ylabel('M/ t2', fontsize=12)
