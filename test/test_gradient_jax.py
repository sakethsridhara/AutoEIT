import context
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import numpy.linalg as npla
import scipy.sparse as spsp
from scipy.sparse.linalg import spsolve
from jax.scipy import optimize
import jax
from jax import value_and_grad

import scipy.optimize as op

import os
import sys

from eit import *  

# boolean for plotting 
plot_bool = False

# loading the file containing the mesh
mat_fname  = 'data/mesh_small.mat'
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

sigma_vec = mat_contents['sigma_vec']

sigma_vec = jnp.array(sigma_vec.reshape((-1)))

dtn_data0 = 0;

swj = SolveWithJax(v_h, dtn_data0)

# this is going to be our referecnes (or data)
dtn_data, sol = swj.dtn_map_jax(sigma_vec)

swj = SolveWithJax(v_h, dtn_data)

# useful for debugging
if plot_bool: 
	# create a triangulation object 
	triangulation = tri.Triangulation(p[:,0], p[:,1], t)
	# plot the triangles
	plt.triplot(triangulation, '-k')
	# plotting the solution 
	plt.tricontourf(triangulation, sol[:,0])
	# plotting a colorbar
	plt.colorbar()
	# show
	plt.show()

# this is the guess
sigma_vec_0 = 1.0 + 0*sigma_vec

sigma_vec_0 = jnp.array(sigma_vec_0)

misfit = swj.misfit_sigma_jax(sigma_vec_0)

# computing misfit and grad
misfit, grad = value_and_grad(swj.misfit_sigma_jax)(sigma_vec_0)

# the reference grad
grad_ref = mat_contents['grad_ref']

err_grad = grad-grad_ref.reshape((-1,))

assert npla.norm(err_grad) < 1.e-6
print("Error with respect to the reference gradient is %.4e" % npla.norm(err_grad))

# otherwise you can use the check_grad function
check_grad = False

# this may be very slow
if check_grad:

	# this is very inefficient
	def misfit_fn(sigma):
		return misfit_sigma(v_h, dtn_data, sigma)[0]

	def grad_fn(sigma):
		return misfit_sigma(v_h, dtn_data, sigma)[1]

	err = op.check_grad(misfit_fn, grad_fn, sigma_vec_0)

	print(err)


# simple optimization routine
def J(x):
	return swj.misfit_sigma_jax(x)

# we define a relatively high tolerance
# recall that this is the square of the misfit
opt_tol = 1.e-6

# running the optimization routine
res = jax.scipy.optimize.minimize(J, sigma_vec_0, method = "BFGS", tol = opt_tol,\
                   options={'maxiter': 1000})

# extracting guess from the resulting optimization 
sigma_guess = res.x

# we proejct sigma back to V in order to plot it
p_v_w = projection_v_w(v_h)
Mass = spsp.csr_matrix(mass_matrix(v_h))

sigma_v = spsolve(Mass, p_v_w@sigma_guess)

# create a triangulation object 
triangulation = tri.Triangulation(p[:,0], p[:,1], t)
# plot the triangles
plt.triplot(triangulation, '-k')
# plotting the solution 
plt.tricontourf(triangulation, sigma_v)
# plotting a colorbar
plt.colorbar()
# show
plt.show()
