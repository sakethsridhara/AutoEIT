import context
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import numpy.linalg as npla
import scipy.sparse as spsp
from scipy.optimize import Bounds
from jax_minimize_wrapper import minimize

from scipy.sparse.linalg import spsolve
from time import perf_counter

import scipy.optimize as op

import jax
from jax import value_and_grad
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)
# from jax.scipy import optimize

import os
import sys

from eit import *  

# boolean for plotting 
plot_bool = False

# loading the file containing the mesh
mat_fname  = 'data/data_reconstruction.mat'
mat_contents = sio.loadmat(mat_fname)

# points
p = mat_contents['p']
#triangle
t = mat_contents['t']-1 # all the indices should be reduced by one

# volumetric indices
vol_idx = mat_contents['vol_idx'].reshape((-1,))-1 # all the indices should be reduced by one
# indices at the boundaries
bdy_idx = mat_contents['bdy_idx'].reshape((-1,))-1 # all the indices should be reduced by one

# define the mesh
mesh = Mesh(p, t, bdy_idx, vol_idx)

# define the approximation space
v_h = V_h(mesh)

# extracting the DtN data
dtn_data = mat_contents['DtN']

# this is the initial guess
sigma_vec_0 = 1 + np.zeros(t.shape[0])
print('---------------START---------------')
swj = SolveWithJax(v_h, dtn_data)

l2_err, grad = value_and_grad(swj.misfit_sigma_jax)(sigma_vec_0)
print('---------------DONE---------------')

assert(np.abs(1.112862432005899e-04-l2_err) < 1.e-7)

# simple optimization routine
def J(x):
 	return swj.misfit_sigma_jax(x)

# simple optimization routine
def J1(x):
    l2_err, grad = value_and_grad(swj.misfit_sigma_jax)(x)
    return np.array(l2_err), np.array(grad)

# we define a relatively high tolerance
# recall that this is the square of the misfit
opt_tol = 1e-9
bounds_l = [1. for _ in range(len(sigma_vec_0))]
bounds_r = [np.inf for _ in range(len(sigma_vec_0))]
bounds = Bounds(bounds_l, bounds_r)

print('---------------START OPT---------------')

#TYPe1 works without bounds
# running the optimization routine

# res = jax.scipy.optimize.minimize(J, sigma_vec_0, method = "l-bfgs-experimental-do-not-rely-on-this", tol = opt_tol,\
#                     options={'maxiter': 10000})


#TYPE 2 #comes from the minimize wrapper
start = perf_counter()

res = minimize(J, sigma_vec_0, method = "L-BFGS-B", bounds = bounds, tol = opt_tol, options={'maxiter': 200,
          'disp': False})

stop = perf_counter()

print("Elapsed time during the OPT in seconds:",stop-start)
# Type -3 Works but slow
# res = op.minimize(J1, sigma_vec_0, method='L-BFGS-B',
#                     jac = True,
#                     tol = opt_tol,
#                     bounds=bounds, 
#                     options={'maxiter': 200,
#                             'disp': True})
print('---------------DONE OPT---------------')

# extracting guess from the resulting optimization 
sigma_guess = res.x

# we proejct sigma back to V in order to plot it
p_v_w = projection_v_w(v_h)
Mass = spsp.csr_matrix(mass_matrix(v_h))

sigma_v = spsolve(Mass, p_v_w@sigma_guess)

# create a triangulation object 
triangulation = tri.Triangulation(p[:,0], p[:,1], t)
# plot the triangles
# plt.triplot(triangulation, '-k')
# plotting the solution 
plt.tricontourf(triangulation, sigma_v)
# plotting a colorbar
plt.colorbar()
# show
plt.show()
