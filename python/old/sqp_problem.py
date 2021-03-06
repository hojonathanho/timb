import numpy as np
import optimization
import time
from interpolation import gradient, jacobian, BilinearSurface, BicubicSurface
import collections
from math import sqrt

class Config(object):
  GRID_NX = 50
  GRID_NY = 50
  GRID_MIN = (-1., -1.)
  GRID_MAX = (1., 1.)

  @classmethod
  def set(cls, grid_nx, grid_ny, grid_min, grid_max):
    cls.GRID_NX, cls.GRID_NY, cls.GRID_MIN, cls.GRID_MAX = grid_nx, grid_ny, grid_min, grid_max
    cls.GRID_SHAPE = (cls.GRID_NX, cls.GRID_NY)
    cls.PIXEL_AREA = (cls.GRID_MAX[0]-cls.GRID_MIN[0])/float(cls.GRID_NX) * (cls.GRID_MAX[1]-cls.GRID_MIN[1])/float(cls.GRID_NY)
    cls.WEIGHT_SCALE = 100

def cleanup_grb_linexpr(e, coeff_zero_cutoff=1e-20):
  var2coeff = collections.defaultdict(lambda: 0.)
  for i in range(e.size()):
    var, coeff = e.getVar(i), e.getCoeff(i)
    if abs(coeff) > coeff_zero_cutoff:
      var2coeff[var] += coeff

  out = e.getConstant()
  for var, coeff in var2coeff.iteritems():
    out += coeff*var
  return out

def make_interp(mat):
  cls = BilinearSurface
  return cls(Config.GRID_MIN[0], Config.GRID_MAX[0], Config.GRID_MIN[1], Config.GRID_MAX[1], mat)

def make_varname_mat(prefix, shape):
  import itertools
  out = np.empty(shape, dtype=object)
  for inds in itertools.product(*[range(d) for d in shape]):
    out[inds] = prefix + '_' + '_'.join(map(str, inds))
  return out

def sumsq(a):
  return (a**2).sum()

# class TestCost(optimization.CostFunc):
#   a = 3
#   def __init__(self, u_vars, coeff):
#     self.u_vars = u_vars
#     self.coeff = coeff
#   def get_vars(self): return self.u_vars
#   def get_name(self): return 'test'
#   def eval(self, vals): return self.coeff * ((vals-self.a)**4).sum()
#   def convex(self, vals): return self.coeff * (( -vals**2 + self.a**2 + (2*vals-2*self.a)*self.u_vars   )**2).sum()


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
  # def eval(self, vals):
  #   J = jacobian(vals, self.eps_x, self.eps_y)
  #   JT = J.transpose((0, 1, 3, 2))
  #   JTJ = (JT[:,:,:,:,None] * J[:,:,None,:,:]).sum(axis=3)
  #   return self.coeff * sumsq(JTJ + J + JT)

  # def convex(self, vals):

  # def convex(self, vals):
  #   J = jacobian(self.u_vars, self.eps_x, self.eps_y)
  #   out = 0
  #   for i in range(J.shape[0]):
  #     for j in range(J.shape[1]):
  #       for k in range(J.shape[2]):
  #         for l in range(J.shape[3]):
  #           term = J[i,j,k,l] + J[i,j,l,k]
  #           out += term*term
  #   out *= self.coeff
  #   return out

class ObservationCost(optimization.CostFunc):
  def __init__(self, phi_vars, obs_pts, coeff):
    self.phi_vars, self.obs_pts, self.coeff = phi_vars, obs_pts, coeff
    self.phi_var_grid = make_interp(self.phi_vars)

  def get_vars(self): return self.phi_vars
  def get_name(self): return 'obs'

  def eval(self, vals):
    surf = make_interp(vals) # TODO: cache
    vals_at_obs = surf.eval_xys(self.obs_pts)
    return self.coeff * sumsq(vals_at_obs)

  def convex(self, vals):
    out = 0
    surf = make_interp(vals) # TODO: cache
    exprs = self.phi_var_grid.eval_xys(self.obs_pts)
    for e in exprs:
      e_clean = cleanup_grb_linexpr(e)
      out += e_clean*e_clean
    out *= self.coeff
    return out

class FlowAgreementCost(optimization.CostFunc):
  def __init__(self, phi_vars, u_vars, prev_phi_surf, coeff):
    self.phi_vars, self.u_vars, self.prev_phi_surf, self.coeff = phi_vars, u_vars, prev_phi_surf, coeff

    self.all_vars = np.empty((Config.GRID_NX, Config.GRID_NY, 3), dtype=object)
    self.all_vars[:,:,0] = self.phi_vars
    self.all_vars[:,:,1:] = self.u_vars

    # self.phi_var_grid = make_interp(self.phi_vars)

  def get_vars(self): return self.all_vars
  def get_name(self): return 'flow_agreement'

  def eval(self, vals):
    phi_vals, u_vals = vals[:,:,0], vals[:,:,1:]
    grid_xys = self.prev_phi_surf.get_grid_xys()
    flowed_prev_phi = self.prev_phi_surf.eval_xys(grid_xys - u_vals.reshape((-1,2))).reshape(Config.GRID_SHAPE)
    return self.coeff * sumsq(phi_vals - flowed_prev_phi)

  def convex(self, vals):
    u0 = vals[:,:,1:]
    points0 = self.prev_phi_surf.get_grid_xys() - u0.reshape((-1, 2))
    consts = self.prev_phi_surf.eval_xys(points0).reshape((Config.GRID_NX, Config.GRID_NY))
    grads = self.prev_phi_surf.grad_xys(points0).reshape((Config.GRID_NX, Config.GRID_NY, 2))

    out = 0
    for i in range(Config.GRID_NX):
      for j in range(Config.GRID_NY):
        diff_u = self.u_vars[i,j,:] - u0[i,j,:]
        e = self.phi_vars[i,j] - float(consts[i,j]) + grads[i,j,:].dot(diff_u)
        out += e*e
    out *= self.coeff
    return out



# class PhiSoftTrustCost(optimization.CostFunc):
#   def __init__(self, phi_vars, prev_phi_vals, coeff):
#     self.phi_vars, self.prev_phi_vals, self.coeff = phi_vars, prev_phi_vals, coeff

#   def get_vars(self): return self.phi_vars
#   def get_name(self): return 'phi_soft_trust'

#   def eval(self, vals):
#     phi_vals = vals
#     assert phi_vals.shape == self.prev_phi_vals.shape
#     return self.coeff * sumsq(phi_vals - self.prev_phi_vals)

#   def convex(self, vals):
#     out = 0
#     for i in range(Config.GRID_NX):
#       for j in range(Config.GRID_NY):
#         e = self.phi_vars[i,j] - float(self.prev_phi_vals[i,j])
#         out += e*e
#     out *= self.coeff
#     return out

# class TPSCost(optimization.CostFunc):
#   def __init__(self, name, field_vars, eps_x, eps_y, coeff):
#     self.name, self.field_vars, self.eps_x, self.eps_y, self.coeff = name, field_vars, eps_x, eps_y, coeff

#   def get_vars(self): return self.field_vars
#   def get_name(self): return self.name

#   def eval(self, vals):
#     # TODO: vectorize, add boundary terms
#     out = 0
#     for i in range(1, Config.GRID_NX-1):
#       for j in range(1, Config.GRID_NY-1):
#         dx = (vals[i-1,j] - 2.*vals[i,j] + vals[i+1,j]) / (self.eps_x**2)
#         out += dx*dx
#         dy = (vals[i,j-1] - 2.*vals[i,j] + vals[i,j+1]) / (self.eps_y**2)
#         out += dy*dy
#         dxy = (vals[i+1,j+1] - vals[i+1,j-1] - vals[i-1,j+1] + vals[i-1,j-1]) / (4.*self.eps_x*self.eps_y)
#         out += 2.*dxy*dxy
#     out *= self.coeff
#     return out

#   def convex(self, vals):
#     out = 0
#     for i in range(1, Config.GRID_NX-1):
#       for j in range(1, Config.GRID_NY-1):
#         dx = (self.field_vars[i-1,j] - 2.*self.field_vars[i,j] + self.field_vars[i+1,j]) / (self.eps_x**2)
#         out += dx*dx
#         dy = (self.field_vars[i,j-1] - 2.*self.field_vars[i,j] + self.field_vars[i,j+1]) / (self.eps_y**2)
#         out += dy*dy
#         dxy = (self.field_vars[i+1,j+1] - self.field_vars[i+1,j-1] - self.field_vars[i-1,j+1] + self.field_vars[i-1,j-1]) / (4.*self.eps_x*self.eps_y)
#         out += 2.*dxy*dxy
#     out *= self.coeff
#     return out

class GradientNormCost(optimization.CostFunc):
  USE_ONE_SIDED = True

  def __init__(self, name, field_vars, eps_x, eps_y, coeff):
    self.name, self.field_vars, self.eps_x, self.eps_y, self.coeff = name, field_vars, eps_x, eps_y, coeff

  def get_vars(self): return self.field_vars
  def get_name(self): return self.name

  def eval(self, vals):
    # TODO: vectorize, add boundary terms
    out = 0.

    if self.USE_ONE_SIDED:
      for i in range(Config.GRID_NX-1):
        for j in range(Config.GRID_NY-1):
          dx = (vals[i+1,j] - vals[i,j]) / self.eps_x
          out += dx*dx
          dy = (vals[i,j+1] - vals[i,j]) / self.eps_y
          out += dy*dy

    else:
      for i in range(1, Config.GRID_NX-1):
        for j in range(1, Config.GRID_NY-1):
          dx = (vals[i+1,j] - vals[i-1,j]) / (2.*self.eps_x)
          out += dx*dx
          dy = (vals[i,j+1] - vals[i,j-1]) / (2.*self.eps_y)
          out += dy*dy

    out *= self.coeff
    return out

  def convex(self, vals):
    out = 0.
    if self.USE_ONE_SIDED:
      for i in range(Config.GRID_NX-1):
        for j in range(Config.GRID_NY-1):
          dx = (self.field_vars[i+1,j] - self.field_vars[i,j]) / self.eps_x
          out += dx*dx
          dy = (self.field_vars[i,j+1] - self.field_vars[i,j]) / self.eps_y
          out += dy*dy

    else:
      for i in range(1, Config.GRID_NX-1):
        for j in range(1, Config.GRID_NY-1):
          dx = (self.field_vars[i+1,j] - self.field_vars[i-1,j]) / (2.*self.eps_x)
          out += dx*dx
          dy = (self.field_vars[i,j+1] - self.field_vars[i,j-1]) / (2.*self.eps_y)
          out += dy*dy

    out *= self.coeff
    return out





# class FlowDivergenceCost(optimization.CostFunc):
#   def __init__(self, u_vars, eps_x, eps_y, coeff):
#     self.u_vars = u_vars
#     self.eps_x, self.eps_y = eps_x, eps_y
#     self.coeff = coeff

#   def get_vars(self): return self.u_vars
#   def get_name(self): return 'flow_div'
#   def eval(self, vals):
#     out = 0
#     for i in range(1, vals.shape[0]-1):
#       for j in range(1, vals.shape[1]-1):
#         e = (vals[i+1,j,0] - vals[i-1,j,0]) / (2.*self.eps_x)
#         e += (vals[i,j+1,1] - vals[i,j-1,1]) / (2.*self.eps_y)
#         out += e*e
#     out *= self.coeff
#     return out

#   def convex(self, vals):
#     out = 0
#     for i in range(1, vals.shape[0]-1):
#       for j in range(1, vals.shape[1]-1):
#         e = (self.u_vars[i+1,j,0] - self.u_vars[i-1,j,0]) / (2.*self.eps_x)
#         e += (self.u_vars[i,j+1,1] - self.u_vars[i,j-1,1]) / (2.*self.eps_y)
#         out += e*e
#     out *= self.coeff
#     return out

class TrackingProblem(object):
  def __init__(self): # TODO: put world bounds here
    self.costs_over_iters = []
    self.opt = optimization.GurobiSQP()

    self.phi_names = make_varname_mat('phi', Config.GRID_SHAPE)
    self.u_names = make_varname_mat('u', Config.GRID_SHAPE + (2,))
    self.phi_vars = self.opt.add_vars(self.phi_names)
    self.u_vars = self.opt.add_vars(self.u_names)
    self.opt.declare_input_var_ordering(np.concatenate((self.phi_vars.ravel(), self.u_vars.ravel())))

    self.prev_phi_surf = make_interp(np.ones(Config.GRID_SHAPE))

    self.flow_norm_cost = FlowNormCost(self.u_vars, None)
    self.opt.add_cost(self.flow_norm_cost)

    self.flow_rigidity_cost = FlowRigidityCost(self.u_vars, self.prev_phi_surf.eps_x, self.prev_phi_surf.eps_y, None)
    self.opt.add_cost(self.flow_rigidity_cost)

    self.obs_cost = ObservationCost(self.phi_vars, None, None)
    self.opt.add_cost(self.obs_cost)

    self.flow_agree_cost = FlowAgreementCost(self.phi_vars, self.u_vars, None, None)
    self.opt.add_cost(self.flow_agree_cost)

    # self.phi_soft_trust_cost = PhiSoftTrustCost(self.phi_vars, np.ones_like(self.phi_vars), 1e-9)
    # self.opt.add_cost(self.phi_soft_trust_cost)

    # self.phi_tps_cost = PhiTPSCost(self.phi_vars, self.prev_phi_surf.eps_x, self.prev_phi_surf.eps_y, 1e-9)
    # self.opt.add_cost(self.phi_tps_cost)

    # self.flow_div_cost = FlowDivergenceCost(self.u_vars, self.prev_phi_surf.eps_x, self.prev_phi_surf.eps_y, 100)
    # self.opt.add_cost(self.flow_div_cost)

    # self.flow_tps_0_cost = TPSCost('flow_tps_0', self.u_vars[:,:,0], self.prev_phi_surf.eps_x, self.prev_phi_surf.eps_y, None)
    # self.opt.add_cost(self.flow_tps_0_cost)
    # self.flow_tps_1_cost = TPSCost('flow_tps_1', self.u_vars[:,:,1], self.prev_phi_surf.eps_x, self.prev_phi_surf.eps_y, None)
    # self.opt.add_cost(self.flow_tps_1_cost)
    self.flow_grad_0_cost = GradientNormCost('flow_grad_0', self.u_vars[:,:,0], self.prev_phi_surf.eps_x, self.prev_phi_surf.eps_y, None)
    self.opt.add_cost(self.flow_grad_0_cost)
    self.flow_grad_1_cost = GradientNormCost('flow_grad_1', self.u_vars[:,:,1], self.prev_phi_surf.eps_x, self.prev_phi_surf.eps_y, None)
    self.opt.add_cost(self.flow_grad_1_cost)

    self.set_coeffs(flow_norm=0, flow_rigidity=10, obs=1, flow_agree=1, flow_tps=1e-5)


  def set_coeffs(self, flow_norm=None, flow_rigidity=None, obs=None, flow_agree=None, flow_tps=None):
    if flow_norm is not None:
      self.flow_norm_cost.coeff = Config.WEIGHT_SCALE * Config.PIXEL_AREA * flow_norm
    if flow_rigidity is not None:
      self.flow_rigidity_cost.coeff = Config.WEIGHT_SCALE*Config.PIXEL_AREA*flow_rigidity
    if obs is not None:
      self.obs_cost.coeff = Config.WEIGHT_SCALE*sqrt(Config.PIXEL_AREA)*obs
    if flow_agree is not None:
      self.flow_agree_cost.coeff = Config.WEIGHT_SCALE*Config.PIXEL_AREA*flow_agree
    if flow_tps is not None:
      self.flow_grad_0_cost.coeff = self.flow_grad_1_cost.coeff = flow_tps

  def set_prev_phi(self, prev_phi):
    s = make_interp(prev_phi)
    self.prev_phi_surf = s
    self.flow_agree_cost.prev_phi_surf = s
    # self.phi_soft_trust_cost.prev_phi_vals = prev_phi

  def set_obs_points(self, obs_pts):
    self.obs_cost.obs_pts = obs_pts

  def optimize(self, init_phi_vals, init_u_vals, return_full=False):
    assert init_u_vals.shape == self.u_vars.shape

    x0 = np.concatenate((init_phi_vals.ravel(), init_u_vals.ravel()))
    result = self.opt.optimize(x0)

    print 'Optimization result:\n', result
    phi_len = self.prev_phi_surf.data.size

    if return_full:
      result_phis, result_us = [], []
      for x in result.x_over_iters:
        result_phis.append(x[:phi_len].reshape(Config.GRID_SHAPE))
        result_us.append(x[phi_len:].reshape(Config.GRID_SHAPE + (2,)))
      return result_phis, result_us, result

    else:
      result_phi, result_u = result.x[:phi_len].reshape(Config.GRID_SHAPE), result.x[phi_len:].reshape(Config.GRID_SHAPE + (2,))
      return result_phi, result_u



  #opt.obj_convex_fn = lambda _: sumsq(phi_vars) + sumsq(u_vars)

  #print opt.optimize(np.ones(num_vars))
  # print opt.get_values(phi_vars)
  # print opt.get_values(u_vars)


if __name__ == '__main__':
  construct_opt()
