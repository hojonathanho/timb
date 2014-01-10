from ctimbpy import *

import numpy as np
from collections import namedtuple

class Coeffs(object):
  flow_norm = 1e-2
  flow_rigidity = 1.
  observation = 1.
  agreement = 1.

Result = namedtuple('Result', ['phi', 'u_x', 'u_y'])

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


class Tracker(object):
  def __init__(self, gp):
    self.gp = gp

    self.opt = Optimizer()
    self.phi_vars = make_var_field(self.opt, 'phi', self.gp)
    self.u_x_vars = make_var_field(self.opt, 'u_x', self.gp)
    self.u_y_vars = make_var_field(self.opt, 'u_y', self.gp)

    self.flow_norm_cost = FlowNormCost(self.u_x_vars, self.u_y_vars)
    self.flow_rigidity_cost = FlowRigidityCost(self.u_x_vars, self.u_y_vars)
    self.observation_cost = ObservationCost(self.phi_vars)
    self.agreement_cost = AgreementCost(self.phi_vars, self.u_x_vars, self.u_y_vars)
    self.opt.add_cost(self.flow_norm_cost, Coeffs.flow_norm)
    self.opt.add_cost(self.flow_rigidity_cost, Coeffs.flow_rigidity)
    self.opt.add_cost(self.observation_cost, Coeffs.observation)
    self.opt.add_cost(self.agreement_cost, Coeffs.agreement)

  def optimize(self, init_state):
    opt_result = self.opt.optimize(init_state.pack())
    # print list(opt_result.cost_over_iters)
    # import matplotlib.pyplot as plt
    # plt.plot(opt_result.cost_over_iters)
    # plt.show()
    result = State.FromPacked(self.gp, opt_result.x)
    # print opt_result.x_over_iters
    return result, opt_result











  # self.observation_cost.set_observation
  # self.agreement_cost.set_prev_phi_and_weights
