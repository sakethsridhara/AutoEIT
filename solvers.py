import numpy as np
import cvxopt
import cvxopt.cholmod
from jax import custom_vjp
from scipy.sparse import coo_matrix
import jax.numpy as jnp
import jax.scipy as jsp

class SparseSolver:
  def __init__(self, meshSpec, bc, iK, jK):
    self.meshSpec = meshSpec
    self.iK, self.jK = iK, jK
    self.bc = bc
    self.sparseSolver = self.initSolver()
  #-----------------------#
  @staticmethod
  def deleterowcol(A, delrow, delcol):
    #Assumes that matrix is in symmetric csc form !
    m = A.shape[0]
    keep = np.delete (np.arange(0, m), delrow)
    A = A[keep, :]
    keep = np.delete (np.arange(0, m), delcol)
    A = A[:, keep]
    return A
  #-----------------------#
  def initSolver(self):
    @custom_vjp
    def solveKuf(Kelem, f):
      ndof = self.meshSpec['ndof']
      u=np.zeros((ndof));
      sK = Kelem.flatten().astype(np.double)

      K = coo_matrix((sK,(self.iK,self.jK)),shape=(ndof,ndof)).tocsc()
      K = self.deleterowcol(K,self.bc['fixed'],self.bc['fixed']).tocoo()
      K = cvxopt.spmatrix(K.data.astype(np.double),K.row.astype(np.int),K.col.astype(np.int))
      B = cvxopt.matrix(f[self.bc['free']])
      cvxopt.cholmod.linsolve(K,B)
      u[self.bc['free']]=np.array(B)[:,0]
      return u

    def solveKuf_fwd(Kelem, f):
      u = solveKuf(Kelem, f)
      return u, (Kelem, u)  # save bounds as residuals

    def solveKuf_bwd(res, g):
      Kelem, u = res
      gradb = solveKuf(Kelem, g)
      gradA = -np.einsum('i,i->i',gradb[np.array(self.iK)], \
                         u[np.array(self.jK)]).\
                         reshape((self.meshSpec['numElems'],8,8))
      return (gradA, gradb)  
    
    solveKuf.defvjp(solveKuf_fwd, solveKuf_bwd)
    return solveKuf
  #-----------------------#
  def solve(self, Kelem):
    def solveKuf(Kelem):
      u = self.sparseSolver(Kelem, self.bc['force'].reshape((-1,1)))
      return u

    u = solveKuf(Kelem)  

    return u  
  #-----------------------#
