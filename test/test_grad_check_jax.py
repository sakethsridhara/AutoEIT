import context
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import numpy.linalg as npla
import scipy.sparse as spsp
from scipy.sparse.linalg import spsolve
import jax
from jax import value_and_grad
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)

from jax.scipy import optimize

import scipy.optimize as op
from time import perf_counter

import os
import sys

from eit import *  

# boolean for plotting 
plot_bool = False

# loading the file containing the mesh
mat_fname  = 'data/data_test.mat'

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

sigma_vec = mat_contents['sigma_vec'].astype(np.float64).reshape((-1,))
sigma_vec_0 = mat_contents['sigma_0'].astype(np.float64).reshape((-1,)) # 661
dtn_data = mat_contents['DtN'].astype(np.float64)

# computing misfit and grad
# misfit, grad = misfit_sigma(v_h, dtn_data, sigma_vec_0)


sigma_vec_0 = jnp.array(sigma_vec_0)
sigma_vec = jnp.array(sigma_vec)


# dtn_data = jnp.array(dtn_data)

print('---------------START---------------')
swj = SolveWithJax(v_h, dtn_data)

# misfit = swj.misfit_sigma_jax(sigma_vec_0)

misfit, grad = value_and_grad(swj.misfit_sigma_jax)(sigma_vec_0)


# swj.check_gradJAX(sigma_vec_0)  # check up to 2nd order derivatives

# the reference grad
grad_ref = mat_contents['grad']
# the reference misfit
misfit_ref = mat_contents['l2']

err_grad = grad-grad_ref.reshape((-1,))

assert np.abs(misfit - misfit_ref) < 1.e-12
print("Error with respect to the reference misfit is %.4e" % np.abs(misfit - misfit_ref))
assert npla.norm(err_grad)/npla.norm(grad_ref.reshape((-1,))) < 1.e-12
print("Error with respect to the reference gradient is %.4e" % npla.norm(err_grad))



check_grad = False

# this may be very slow
if check_grad:

	# this is very inefficient
	def misfit_fn(sigma):
		return swj.misfit_sigma_jax(sigma)

	def grad_fn(sigma):
		return grad(swj.misfit_sigma_jax)(sigma)

	err = op.check_grad(misfit_fn, grad_fn, sigma_vec_0)

	print("Error of the gradient wrt FD approximation %.4e" %err)




dtn, sol = swj.dtn_map_jax(sigma_vec)


# 
# dtn, sol = dtn_map(v_h, sigma_vec)

dtn_ref = mat_contents['DtN']	

err_DtN_vec = dtn - dtn_ref
err_DtN = npla.norm(err_DtN_vec.reshape((-1,)))/npla.norm(dtn_ref.reshape((-1,)))

assert err_DtN < 1.e-12
print("Error of the DtN maps with respect to reference %.4e" % err_DtN)




# simple optimization routine
def J(x):
	return swj.misfit_sigma_jax(x)


opt_tol = 1.e-6

print('---------------START OPT---------------')
start = perf_counter()
# running the optimization routine
res = jax.scipy.optimize.minimize(J, sigma_vec_0, method = "BFGS", tol = opt_tol,\
                   options={'maxiter': 500})
stop = perf_counter()
print("Elapsed time during the OPT in seconds:",stop-start)
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
plt.triplot(triangulation, '-k')
# plotting the solution 
plt.tricontourf(triangulation, sigma_v)
# plotting a colorbar
plt.colorbar()
# show
plt.show()
