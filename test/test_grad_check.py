import context
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import numpy.linalg as npla
import scipy.sparse as spsp
from scipy.sparse.linalg import spsolve

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

sigma_vec = mat_contents['sigma_vec'].astype(np.float64)
sigma_vec_0 = mat_contents['sigma_0'].astype(np.float64).reshape((-1,))

dtn_data = mat_contents['DtN'].astype(np.float64)

# computing misfit and grad
misfit, grad = misfit_sigma(v_h, dtn_data, sigma_vec_0)

# the reference grad
grad_ref = mat_contents['grad']

# the reference misfit
misfit_ref = mat_contents['l2']

err_grad = grad-grad_ref.reshape((-1,))

assert npla.norm(err_grad)/npla.norm(grad_ref.reshape((-1,))) < 1.e-12
print("Error with respect to the reference gradient is %.4e" % npla.norm(err_grad))

assert np.abs(misfit - misfit_ref) < 1.e-12
print("Error with respect to the reference misfit is %.4e" % np.abs(misfit - misfit_ref))

# 
dtn, sol = dtn_map(v_h, sigma_vec)

dtn_ref = mat_contents['DtN']	

err_DtN_vec = dtn - dtn_ref
err_DtN = npla.norm(err_DtN_vec.reshape((-1,)))/npla.norm(dtn_ref.reshape((-1,)))

assert err_DtN < 1.e-12
print("Error of the DtN maps with respect to reference %.4e" % err_DtN)


def misfit_simple(v_h, dtn_data, sigma_vec):
	# compute dtn and sol for given sigma
    dtn, sol = dtn_map(v_h, sigma_vec)

    # compute the residual
    residual  = -(dtn_data - dtn)

    return 0.5*np.sum(np.square(residual))

check_grad = True

# this may be very slow
if check_grad:

	# this is very inefficient
	def misfit_fn(sigma):
		return misfit_simple(v_h, dtn_data, sigma)

	def grad_fn(sigma):
		return misfit_sigma(v_h, dtn_data, sigma)[1]

	err = op.check_grad(misfit_fn, grad_fn, sigma_vec_0)

	print("Error of the gradient wrt FD approximation %.4e" %err)


# simple optimization routine
def J(x):
	return misfit_sigma(v_h, dtn_data, x)

# we define a relatively high tolerance
# recall that this is the square of the misfit
opt_tol = 1.e-6
start = perf_counter()

# running the optimization routine
res = op.minimize(J, sigma_vec_0, #method='L-BFGS-B',
                   jac = True,
                   options={'eps': opt_tol, 
                   			'maxiter': 500,
                   			'disp': False})
stop = perf_counter()
print("Elapsed time during the OPT in seconds:",stop-start)

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
