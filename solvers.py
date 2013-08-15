import numpy as np

from grid import grid_interp_grad

class Weights:
  flow_norm = .01#.0001
  flow = 1.
  rigidity = 10.
  obs = 1.

def _eval_cost(pixel_area, obs_n2, prev_sdf, sdf, u, ignore_obs=False, return_full=False):
  ''' everything in world coordinates '''

  total = 0.
  costs = {}

  # small flow
  flow_cost = Weights.flow_norm * pixel_area * (u.data()**2).sum()
  costs['flow_norm'] = flow_cost
  total += flow_cost

  # smooth flow (small gradients)
  # flow_smooth_cost = pixel_area * (u.jac_data()**2).sum()
  # costs['flow_smooth'] = flow_smooth_cost
  # total += flow_smooth_cost

  #XXXXXXXXXXXXXXXXXXXXXXXX
  #import IPython; IPython.embed()
  # # sdf and flow agree
  # shifted_sdf = prev_sdf.flow(u)
  # agree_cost = ((shifted_sdf.data() - sdf.data())**2).sum() * pixel_area
  # # print 'agree', agree_cost
  # costs['agree'] = agree_cost
  # total += agree_cost
  #XXXXXXXXXXXXXXXXXXXXXXXXXXX

  # linearized optical flow
  agree_cost = Weights.flow * pixel_area * (((prev_sdf.jac_data() * u.data()).sum(axis=2) + sdf.data() - prev_sdf.data())**2).sum()
  costs['flow'] = agree_cost
  total += agree_cost

  # rigidity of flow
  ### FINITE STRAIN
  # J = (u.jac_data() + np.eye(2)[None,None,:,:]).reshape((-1, 2, 2))
  # JT = np.transpose(J, axes=(0, 2, 1))
  # products = (JT[:,:,:,None] * J[:,None,:,:]).sum(axis=2)
  # rigid_cost = pixel_area * ((products - np.eye(2)[None,:,:])**2).sum()
  ### INFINITESIMAL STRAIN

  J = u.jac_data()
  JT = np.transpose(J, axes=(0, 1, 3, 2))
  M = J + JT
  rigid_cost = Weights.rigidity * pixel_area * (M**2).sum()
  costs['rigidity'] = rigid_cost
  total += rigid_cost

  # sdf is zero at observation points
  if not ignore_obs:
    sdf_at_obs = sdf.eval_xys(obs_n2)
    obs_cost = Weights.obs * np.sqrt(pixel_area) * (sdf_at_obs**2).sum()
    costs['obs'] = obs_cost
    total += obs_cost

  if return_full:
    return total, costs
  return total


import time
def _eval_cost_grad(pixel_area, obs_n2, prev_sdf, sdf, u, ignore_obs=False):
  t_start = time.time()

  grad = np.zeros(sdf.size() + u.size())
  grad_sdf, grad_u = grad[:sdf.size()], grad[sdf.size():]

  flow_cost_grad_u = Weights.flow_norm * pixel_area * 2.*u.data().ravel()
  grad_u += flow_cost_grad_u

  agree_cost = (prev_sdf.jac_data() * u.data()).sum(axis=2) + sdf.data() - prev_sdf.data()
  agree_cost_grad_sdf = Weights.flow * pixel_area * 2. * agree_cost.ravel()
  agree_cost_grad_u = Weights.flow * pixel_area * 2. * (agree_cost[:,:,None] * prev_sdf.jac_data()).ravel()
  grad_sdf += agree_cost_grad_sdf
  grad_u += agree_cost_grad_u

  J = u.jac_data()
  JT = np.transpose(J, axes=(0, 1, 3, 2))
  M = J + JT
  rigid_cost_grad_u = np.zeros(u.shape())
  # alternative loop implementation
  # for i in range(u.shape()[0]):
  #   for j in range(u.shape()[1]):
  #     A = 0. if i == 0 else M[i-1,j,0,0]
  #     B = 0. if i == u.shape()[0]-1 else M[i+1,j,0,0]
  #     C = 0. if j == 0 else M[i,j-1,0,1]
  #     D = 0. if j == u.shape()[1]-1 else M[i,j+1,0,1]
  #     z = 2. / (u.shape()[0] - 1.) * (u.xmax - u.xmin)
  #     a = (2. if i == 1 else 1.)*A
  #     b = (2. if i == u.shape()[0]-2 else 1.)*B
  #     c = (2. if j == 1 else 1.)*C
  #     d = (2. if j == u.shape()[1]-2 else 1.)*D
  #     e = 0
  #     if i == 0:
  #       e += -2.*M[i,j,0,0]
  #     elif i == u.shape()[0]-1:
  #       e += 2.*M[i,j,0,0]
  #     if j == 0:
  #       e += -2.*M[i,j,0,1]
  #     elif j == u.shape()[0]-1:
  #       e += 2.*M[i,j,0,1]
  #     rigid_cost_grad_u[i,j,0] = 4./z * (a - b + c - d + e)
  #     A = 0. if i == 0 else M[i-1,j,0,1]
  #     B = 0. if i == u.shape()[0]-1 else M[i+1,j,0,1]
  #     C = 0. if j == 0 else M[i,j-1,1,1]
  #     D = 0. if j == u.shape()[1]-1 else M[i,j+1,1,1]
  #     z = 2. / (u.shape()[1] - 1.) * (u.ymax - u.ymin)
  #     a = (2. if i == 1 else 1.)*A
  #     b = (2. if i == u.shape()[0]-2 else 1.)*B
  #     c = (2. if j == 1 else 1.)*C
  #     d = (2. if j == u.shape()[1]-2 else 1.)*D
  #     e = 0
  #     if i == 0:
  #       e += -2.*M[i,j,0,1]
  #     elif i == u.shape()[0]-1:
  #       e += 2.*M[i,j,0,1]
  #     if j == 0:
  #       e += -2.*M[i,j,1,1]
  #     elif j == u.shape()[0]-1:
  #       e += 2.*M[i,j,1,1]
  #     rigid_cost_grad_u[i,j,1] = 4./z * (a - b + c - d + e)
  A = np.zeros((u.shape()[0], u.shape()[1]))
  B = np.zeros((u.shape()[0], u.shape()[1]))
  C = np.zeros((u.shape()[0], u.shape()[1]))
  D = np.zeros((u.shape()[0], u.shape()[1]))
  E = np.zeros((u.shape()[0], u.shape()[1]))
  A[1:,:] = M[:-1,:,0,0]; A[1,:] *= 2.
  B[:-1,:] = M[1:,:,0,0]; B[-2,:] *= 2.
  C[:,1:] = M[:,:-1,0,1]; C[:,1] *= 2.
  D[:,:-1] = M[:,1:,0,1]; D[:,-2] *= 2.
  E[0,:] = -2.*M[0,:,0,0]
  E[-1,:] = 2.*M[-1,:,0,0]
  E[:,0] += -2.*M[:,0,0,1]
  E[:,-1] += 2.*M[:,-1,0,1]
  rigid_cost_grad_u[:,:,0] = 2.*(u.shape()[0] - 1.)/(u.xmax - u.xmin) * (A - B + C - D + E)
  A.fill(0.); B.fill(0.); C.fill(0.); D.fill(0.); E.fill(0.)
  A[1:,:] = M[:-1,:,0,1]; A[1,:] *= 2.
  B[:-1,:] = M[1:,:,0,1]; B[-2,:] *= 2.
  C[:,1:] = M[:,:-1,1,1]; C[:,1] *= 2.
  D[:,:-1] = M[:,1:,1,1]; D[:,-2] *= 2.
  E[0,:] = -2.*M[0,:,0,1]
  E[-1,:] = 2.*M[-1,:,0,1]
  E[:,0] += -2.*M[:,0,1,1]
  E[:,-1] += 2.*M[:,-1,1,1]
  rigid_cost_grad_u[:,:,1] = 2.*(u.shape()[1] - 1.)/(u.ymax - u.ymin) * (A - B + C - D + E)
  rigid_cost_grad_u *= Weights.rigidity * pixel_area
  grad_u += rigid_cost_grad_u.ravel()

  if not ignore_obs:
    sdf_at_obs = sdf.eval_xys(obs_n2)
    obs_cost_grad_sdf = Weights.obs * np.sqrt(pixel_area) * 2. * (sdf_at_obs[:,None,None] * grid_interp_grad(sdf.data(), np.c_[sdf.to_grid_inds(obs_n2[:,0], obs_n2[:,1])])).sum(axis=0).ravel()
    grad_sdf += obs_cost_grad_sdf


  #print 'grad eval time: %f, grad norm: %f' % (time.time()-t_start, np.linalg.norm(grad))

  return grad


class GradientDescentSolver(object):
  def __init__(self):
    pass



import gurobipy
class GurobiSolver(object):
  def __init__(self):
    pass
