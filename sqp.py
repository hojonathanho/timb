import numpy as np
import probimization
import time
from bicubic import gradient, jacobian, BicubicSurface

GRID_NX = 50
GRID_NY = 50
GRID_SHAPE = (GRID_NX, GRID_NY)
GRID_MIN = (-1., -1.)
GRID_MAX = (1., 1.)

def make_bicubic(mat):
  return BicubicSurface(GRID_MIN[0], GRID_MAX[0], GRID_MIN[1], GRID_MAX[1], mat)

def make_varname_mat(prefix, shape):
  import itertools
  out = np.empty(shape, dtype=object)
  for inds in itertools.product(*[range(d) for d in shape]):
    out[inds] = prefix + '_' + '_'.join(map(str, inds))
  return out

def sumsq(a):
  return (a**2).sum()

class TestCost(probimization.CostFunc):
  a = 3
  def __init__(self, u_vars, coeff):
    self.u_vars = u_vars
    self.coeff = coeff
  def get_vars(self): return self.u_vars
  def get_name(self): return 'test'
  def eval(self, vals): return self.coeff * ((vals-self.a)**4).sum()
  def convex(self, vals): return self.coeff * (( -vals**2 + self.a**2 + (2*vals-2*self.a)*self.u_vars   )**2).sum()


class FlowNormCost(probimization.CostFunc):
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

class FlowRigidityCost(probimization.CostFunc):
  def __init__(self, u_vars, eps_x, eps_y, coeff):
    self.u_vars = u_vars
    self.eps_x, self.eps_y = eps_x, eps_y
    self.coeff = coeff

  def get_vars(self): return self.u_vars
  def get_name(self): return 'flow_rigidity'
  def eval(self, vals):
    J = jacobian(vals, self.eps_x, self.eps_y)
    JT = J.transpose((0, 1, 3, 2))
    return self.coeff * sumsq(J + JT)

  def convex(self, vals):
    J = jacobian(self.u_vars, self.eps_x, self.eps_y)
    out = 0
    for i in range(J.shape[0]):
      for j in range(J.shape[1]):
        for k in range(J.shape[2]):
          for l in range(J.shape[3]):
            term = J[i,j,k,l] + J[i,j,l,k]
            out += term*term
    out *= self.coeff
    return out

class ObservationCost(probimization.CostFunc):
  def __init__(self, phi_vars, obs_pts, coeff):
    self.phi_vars, self.obs_pts, self.coeff = phi_vars, obs_pts, coeff
    self.phi_var_grid = make_bicubic(self.phi_vars)

  def get_vars(self): return self.phi_vars
  def get_name(self): return 'obs'

  def eval(self, vals):
    surf = make_bicubic(vals)
    vals_at_obs = surf.eval_xys(self.obs_pts)
    return self.coeff * sumsq(vals_at_obs)

  def convex(self, vals):
    out = 0
    surf = make_bicubic(vals) # TODO: cache
    exprs = self.phi_var_grid.eval_xys(self.obs_pts)
    for e in exprs:
      out += e*e
    out *= self.coeff
    return out

class FlowAgreementCost(probimization.CostFunc):
  def __init__(self, phi_vars, u_vars, prev_phi_surf, coeff):
    self.phi_vars, self.u_vars, self.prev_phi_surf, self.coeff = phi_vars, u_vars, prev_phi_surf, coeff

    self.all_vars = np.empty((GRID_NX, GRID_NY, 3), dtype=object)
    self.all_vars[:,:,0] = self.phi_vars
    self.all_vars[:,:,1:] = self.u_vars

  def get_vars(self): return self.all_vars
  def get_name(self): return 'flow_agreement'

  def eval(self, vals):
    phi_vals, u_vals = vals[:,:,0], vals[:,:,1:]
    xys = self.prev_phi_surf.get_grid_xys()
    flowed_prev_phi = self.prev_phi_surf.eval_xys(xys - u_vals.reshape((-1,2))).reshape(GRID_SHAPE)
    return self.coeff * sumsq(phi_vals - flowed_prev_phi)

  def convex(self, vals):
    u0 = vals[:,:,1:]
    points0 = self.prev_phi_surf.get_grid_xys() - u0.reshape((-1, 2))
    consts = self.prev_phi_surf.eval_xys(points0).reshape((GRID_NX, GRID_NY))
    grads = self.prev_phi_surf.grad_xys(points0).reshape((GRID_NX, GRID_NY, 2))

    out = 0
    for i in range(GRID_NX):
      for j in range(GRID_NY):
        diff_u = self.u_vars[i,j,:] - u0[i,j,:]
        e = self.phi_vars[i,j] - float(consts[i,j]) - grads[i,j,:].dot(diff_u)
        out += e*e
    out *= self.coeff
    return out

def construct_prob():
  prob = probimization.GurobiSQP()

  t_start = time.time() 
  phi_names = make_varname_mat('phi', GRID_SHAPE)
  u_names = make_varname_mat('u', GRID_SHAPE + (2,))
  phi_vars = prob.add_vars(phi_names)
  u_vars = prob.add_vars(u_names)

  prev_phi_surf = make_bicubic(np.ones(GRID_SHAPE))

  prob.add_cost(FlowNormCost(u_vars, 1))
  prob.add_cost(FlowRigidityCost(u_vars, prev_phi_surf.eps_x, prev_phi_surf.eps_y, 1))
  obs_pts = np.array([[1, 2], [3, 4]])
  prob.add_cost(ObservationCost(phi_vars, obs_pts, 1))

  prob.add_cost(FlowAgreementCost(phi_vars, u_vars, prev_phi_surf, 1))

  #prob.obj_convex_fn = lambda _: sumsq(phi_vars) + sumsq(u_vars)

  print 'built problem in', time.time()-t_start
  num_vars = phi_vars.size + u_vars.size
  print prob.probimize(np.ones(num_vars))
  # print prob.get_values(phi_vars)
  # print prob.get_values(u_vars)

if __name__ == '__main__':
  construct_prob()
