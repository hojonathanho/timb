import numpy as np
import optimization
import time
from bicubic import gradient
import gurobipy

GRID_NX = 50
GRID_NY = 50
GRID_SHAPE = (GRID_NX, GRID_NY)

def make_varname_mat(prefix, shape):
  import itertools
  out = np.empty(shape, dtype=object)
  for inds in itertools.product(*[range(d) for d in shape]):
    out[inds] = prefix + '_' + '_'.join(map(str, inds))
  return out

def sumsq(a):
  return (a**2).sum()

def jac(u, eps_x=1, eps_y=1):
  out = np.empty((u.shape[0], u.shape[1], 2, 2), dtype=u.dtype)
  out[:,:,0,:] = gradient(u[:,:,0], eps_x, eps_y, wrt='xy')
  out[:,:,1,:] = gradient(u[:,:,1], eps_x, eps_y, wrt='xy')
  return out

class TestCost(optimization.CostFunc):
  a = 3
  def __init__(self, u_vars, coeff):
    self.u_vars = u_vars
    self.coeff = coeff
  def get_vars(self): return self.u_vars
  def get_name(self): return 'test'
  def eval(self, vals): return self.coeff * ((vals-self.a)**4).sum()
  def convex(self, vals): return self.coeff * (( -vals**2 + self.a**2 + (2*vals-2*self.a)*self.u_vars   )**2).sum()


class FlowNormCost(optimization.CostFunc):
  def __init__(self, u_vars, coeff):
    self.u_vars = u_vars
    self.coeff = coeff
  def get_vars(self): return self.u_vars
  def get_name(self): return 'flow_norm'
  def eval(self, vals):
    return self.coeff * sumsq(vals)
  def convex(self, vals):
    out = 0
    for v in self.u_vars.flat:
      out += v*v
    out *= self.coeff
    return out

class FlowRigidityCost(optimization.CostFunc):
  def __init__(self, u_vars, coeff):
    self.u_vars = u_vars
    self.coeff = coeff

  def get_vars(self): return self.u_vars
  def get_name(self): return 'flow_rigidity'
  def eval(self, vals):
    J = jac(vals)
    JT = J.transpose((0, 1, 3, 2))
    return self.coeff * sumsq(J + JT)

  def convex(self, vals):
    J = jac(self.u_vars)
    out = 0
    for i in range(J.shape[0]):
      for j in range(J.shape[1]):
        for k in range(J.shape[2]):
          for l in range(J.shape[3]):
            term = J[i,j,k,l] + J[i,j,l,k]
            out += term*term
    out *= self.coeff
    return out

class ObservationCost(optimization.CostFunc):
  def __init__(self, phi_vars, obs, coeff):
    self.phi_vars, self.obs, self.coeff = phi_vars, obs, coeff

  def get_vars(self): return self.phi_vars
  def get_name(self): return 'obs'

  def eval(self, vals):
    bicubic.

  def convex(self, vals):
    pass


def construct_prob():
  opt = optimization.GurobiSQP(None)

  t_start = time.time() 
  phi_names = make_varname_mat('phi', GRID_SHAPE)
  u_names = make_varname_mat('u', GRID_SHAPE + (2,))
  phi_vars = opt.add_vars(phi_names)
  u_vars = opt.add_vars(u_names)

  num_vars = phi_vars.size + u_vars.size

  opt.add_cost(FlowNormCost(u_vars, 1))
  opt.add_cost(FlowRigidityCost(u_vars, 1))

  #opt.obj_convex_fn = lambda _: sumsq(phi_vars) + sumsq(u_vars)

  print 'built problem in', time.time()-t_start
  print opt.optimize(np.ones(num_vars))
  # print opt.get_values(phi_vars)
  # print opt.get_values(u_vars)

if __name__ == '__main__':
  construct_prob()
