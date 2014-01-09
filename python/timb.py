from ctimbpy import *

import numpy as np
from collections import namedtuple

class Coeffs(object):
  def __init__(self):
    self.flow_norm
    self.flow_rigidity
    self.observation
    self.agreement



Result = namedtuple('Result', ['phi', 'u_x', 'u_y'])



struct TrackingProblemResult {
  DoubleField phi;
  DoubleField u_x, u_y;
  DoubleField next_phi, next_omega;
  OptResultPtr opt_result;
  TrackingProblemResult(const GridParams& gp) : phi(gp), u_x(gp), u_y(gp), next_phi(gp), next_omega(gp) { }
};
typedef boost::shared_ptr<TrackingProblemResult> TrackingProblemResultPtr;



class State(object):
  def __init__(self, phi, u_x, u_y):
    self.phi, self.u_x, self.u_y = phi, u_x, u_y

  @staticmethod
  def FromPacked(x):
    n = gp.nx * gp.ny # num grid points
    x = x.squeeze()
    assert x.size == 3*n
    phi = x[:n].reshape((gp.nx, gp.ny))
    u_x = x[n:2*n].reshape((gp.nx, gp.ny))
    u_y = x[2*n:].reshape((gp.nx, gp.ny))
    return State(phi, u_x, u_y)

  def pack(self):
    return np.r_[self.phi.ravel(), self.u_x.ravel(), self.u_y.ravel()]


class Tracker(object):
  def __init__(self, gp):
    self.gp = gp

    self.opt = Optimizer()
    self.phi_vars = make_var_field(opt, 'phi', self.gp)
    self.u_x_vars = make_var_field(opt, 'u_x', self.gp)
    self.u_y_vars = make_var_field(opt, 'u_y', self.gp)

    self.flow_norm_cost = FlowNormCost(self.u_x_vars, self.u_y_vars)
    self.flow_rigidity_cost = FlowRigidityCost(self.u_x_vars, self.u_y_vars)
    self.observation_cost = ObservationCost(self.phi_vars)
    self.agreement_cost = AgreementCost(self.phi_vars, self.u_x_vars, self.u_y_vars)
    self.opt.add_cost(self.flow_norm_cost, Coeffs.flow_norm)
    self.opt.add_cost(self.flow_rigidity_cost, Coeffs.flow_rigidity)
    self.opt.add_cost(self.observation_cost, Coeffs.observation)
    self.opt.add_cost(self.agreement_cost, Coeffs.agreement)

  def optimize(self, init_state):
    result = self.opt.optimize(init_state.pack())











  # self.observation_cost.set_observation
  # self.agreement_cost.set_prev_phi_and_weights
