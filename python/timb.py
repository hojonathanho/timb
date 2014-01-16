from ctimbpy import *

import numpy as np
import observation

class Coeffs(object):
  flow_norm = 1e-2
  flow_rigidity = 1.
  observation = 1.
  agreement = 1.


class State(object):
  def __init__(self, gp, phi, u_x, u_y):
    self.gp, self.phi, self.u_x, self.u_y = gp, phi, u_x, u_y

  @staticmethod
  def FromPacked(gp, x):
    n = gp.nx * gp.ny # num grid points
    x = x.squeeze()
    assert x.size == 3*n
    phi = x[:n].reshape((gp.nx, gp.ny))
    u_x = x[n:2*n].reshape((gp.nx, gp.ny))
    u_y = x[2*n:].reshape((gp.nx, gp.ny))
    return State(gp, phi, u_x, u_y)

  def pack(self):
    return np.r_[self.phi.ravel(), self.u_x.ravel(), self.u_y.ravel()]


class Tracker(object):
  def __init__(self, grid_params, use_iterative_reweighting=True):
    self.gp = grid_params
    self.use_iterative_reweighting = use_iterative_reweighting

    self.opt = Optimizer()
    self.phi_vars = make_var_field(self.opt, 'phi', self.gp)
    self.u_x_vars = make_var_field(self.opt, 'u_x', self.gp)
    self.u_y_vars = make_var_field(self.opt, 'u_y', self.gp)

    self.flow_norm_cost = FlowNormCost(self.u_x_vars, self.u_y_vars)
    self.opt.add_cost(self.flow_norm_cost, Coeffs.flow_norm)

    self.flow_rigidity_cost = FlowRigidityCost(self.u_x_vars, self.u_y_vars)
    self.opt.add_cost(self.flow_rigidity_cost, Coeffs.flow_rigidity)

    self.observation_zc_cost = ObservationZeroCrossingCost(self.phi_vars)
    self.opt.add_cost(self.observation_zc_cost, 0)

    self.observation_cost = ObservationCost(self.phi_vars)
    self.opt.add_cost(self.observation_cost, Coeffs.observation)

    self.agreement_cost = AgreementCost(self.phi_vars, self.u_x_vars, self.u_y_vars)
    self.opt.add_cost(self.agreement_cost, Coeffs.agreement)

    self.prev_weights = None

  def set_prev_phi_and_weights(self, prev_phi, weights):
    self.agreement_cost.set_prev_phi_and_weights(prev_phi, weights)
    self.prev_phi, self.prev_weights = prev_phi, weights

  def set_observation(self, obs, weights):
    self.observation_cost.set_observation(obs, weights)
    self.opt.set_cost_coeff(self.observation_cost, Coeffs.observation)
    self.opt.set_cost_coeff(self.observation_zc_cost, 0)

  # def set_observation_zc(self, pts):
  #   self.observation_zc_cost.set_zero_points(pts)
  #   self.opt.set_cost_coeff(self.observation_zc_cost, Coeffs.observation)
  #   self.opt.set_cost_coeff(self.observation_cost, 0)

  def optimize(self, init_state):

    def _optimize_once(state):
      opt_result = self.opt.optimize(state.pack())
      result = State.FromPacked(self.gp, opt_result.x)
      return result, opt_result

    if not self.use_iterative_reweighting:
      return _optimize_once(init_state)

    curr_state, results, opt_results = init_state, [], []
    for i in range(5):
      state, opt_result = _optimize_once(curr_state)

      flowed_prev_weights = apply_flow(self.gp, self.prev_weights, state.u_x, state.u_y)
      self.set_prev_phi_and_weights(self.prev_phi, flowed_prev_weights) # prev_phi stays the same, only weights change

      results.append(state)
      opt_results.append(opt_result)
      curr_state = state

    return results[-1], opt_results[-1]

def reintegrate(phi, ignore_mask):
  from timb_skfmm import distance
  d = distance(phi, ignore_mask=ignore_mask, order=1)
  return np.clip(d, -observation.TRUNC_DIST, observation.TRUNC_DIST)


# Utility functions

def plot_state(state):
  import matplotlib
  import matplotlib.pyplot as plt
  plt.clf()
  matplotlib.rcParams.update({'font.size': 8})

  TSDF_TRUNC = 3.
  plt.subplot(121)
  plt.title('phi')
  plt.axis('off')
  plt.imshow(np.flipud(state.phi.T), vmin=-TSDF_TRUNC, vmax=TSDF_TRUNC, cmap='bwr').set_interpolation('nearest')

  plt.subplot(122, aspect='equal')
  plt.title('u')
  def plot_flow(u_x, u_y):
    assert u_x.shape == u_y.shape
    x = np.linspace(state.gp.xmin, state.gp.xmax, state.gp.nx)
    y = np.linspace(state.gp.ymin, state.gp.ymax, state.gp.ny)
    Y, X = np.meshgrid(x, y)
    plt.quiver(X, Y, u_x, u_y, angles='xy', scale_units='xy', scale=1.)
  plot_flow(state.u_x, state.u_y)

  plt.show()
