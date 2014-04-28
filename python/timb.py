import ctimb
from ctimb import *

def isEnumType(o):
    return isinstance(o, type) and issubclass(o,int) and not (o is int)

def _tuple2enum(enum, value):
    enum = getattr(ctimb, enum)
    e = enum.values.get(value,None)
    if e is None:
        e = enum(value)
    return e

def _registerEnumPicklers(): 
    from copy_reg import constructor, pickle
    def reduce_enum(e):
        enum = type(e).__name__.split('.')[-1]
        return ( _tuple2enum, ( enum, int(e) ) )
    constructor( _tuple2enum)
    for e in [ e for e in vars(ctimb).itervalues() if isEnumType(e) ]:
        pickle(e, reduce_enum)

_registerEnumPicklers()


import numpy as np
import observation


class TrackerParams(object):
  def __init__(self):
    # Optimization parameters
    self.flow_norm_coeff = 1e-6
    self.flow_rigidity_coeff = 1.
    self.observation_coeff = 1.
    self.agreement_coeff = 1.

    self.reweighting_iters = 5
    self.max_inner_iters = 10
    self.lin_solver_iters = 10

    # Observation parameters
    self.tsdf_trunc_dist = 10.
    self.sensor_mode = 'accurate'

    self.obs_weight_epsilon = 0.
    self.obs_weight_delta = 5.
    self.obs_weight_filter_radius = 20
    self.obs_weight_far = True

    self.use_linear_downweight = True
    self.use_min_to_combine = True

    # Smoothing parameters
    self.enable_smoothing = True
    self.smoother_phi_ignore_thresh = self.tsdf_trunc_dist / 2.
    self.smoother_weight_ignore_thresh = 1e-2
    self.smoother_post_fmm = False # whether or not to reinitialize to a SDF after TPS smoothing



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


class TrackingOptimizationProblem(object):
  def __init__(self, grid_params, params):
    assert isinstance(grid_params, GridParams) and isinstance(params, TrackerParams)
    self.gp, self.params = grid_params, params

    self.opt = Optimizer()

    def cb(old_x, delta_x, true_old_cost, true_improvement, model_improvement, ratio):
      print '===================CALLBACK==================='
      print old_x, delta_x, true_old_cost, true_improvement, model_improvement, ratio
      old_state = State.FromPacked(self.gp, old_x)
      print 'Old cost from symbolic:', true_old_cost
      print 'Old cost from hardcoded:', timb_problem_eval_objective(
        self.gp,
        old_state.phi, old_state.u_x, old_state.u_y,
        self.obs, self.obs_weights,
        self.prev_phi, self.curr_wtilde,
        self.params.flow_rigidity_coeff, self.params.flow_norm_coeff
      )
      print '===================END CALLBACK==================='
    self.opt.add_intermediate_callback(cb)

    self.phi_vars = make_var_field(self.opt, 'phi', self.gp)
    self.u_x_vars = make_var_field(self.opt, 'u_x', self.gp)
    self.u_y_vars = make_var_field(self.opt, 'u_y', self.gp)

    self.flow_norm_cost = FlowNormCost(self.u_x_vars, self.u_y_vars)
    self.opt.add_cost(self.flow_norm_cost, self.params.flow_norm_coeff)

    self.flow_rigidity_cost = FlowRigidityCost(self.u_x_vars, self.u_y_vars)
    self.opt.add_cost(self.flow_rigidity_cost, self.params.flow_rigidity_coeff)

    self.observation_cost = ObservationCost(self.phi_vars)
    self.opt.add_cost(self.observation_cost, self.params.observation_coeff)

    self.agreement_cost = AgreementCost(self.phi_vars, self.u_x_vars, self.u_y_vars)
    self.opt.add_cost(self.agreement_cost, self.params.agreement_coeff)

    self.prev_weights = None

  def set_prev_phi_and_weights(self, prev_phi, weights):
    self.agreement_cost.set_prev_phi_and_weights(prev_phi, weights)
    self.prev_phi, self.prev_weights = prev_phi, weights
    self.curr_wtilde = weights

  def set_observation(self, obs, weights):
    self.observation_cost.set_observation(obs, weights)
    self.obs, self.obs_weights = obs, weights

  def optimize(self, init_phi, init_u_x, init_u_y):
    def _optimize_once(state):
      opt_result = self.opt.optimize(state.pack())
      result = State.FromPacked(self.gp, opt_result['x'])
      return result, opt_result

    assert self.params.reweighting_iters >= 1

    init_state = State(self.gp, init_phi, init_u_x, init_u_y)

    if self.params.reweighting_iters == 1:
      return _optimize_once(init_state)

    curr_state, results, opt_results = init_state, [], []
    for i in range(self.params.reweighting_iters):
      state, opt_result = _optimize_once(curr_state)

      flowed_prev_weights = apply_flow_to_weights(self.gp, self.prev_weights, state.u_x, state.u_y)
      self.curr_wtilde = flowed_prev_weights
      self.agreement_cost.set_prev_phi_and_weights(self.prev_phi, flowed_prev_weights) # prev_phi stays the same, only weights change

      results.append(state)
      opt_results.append(opt_result)
      curr_state = state

    print results[-1].u_x
    return results[-1], opt_results[-1]


class TrackingOptimizationProblem2(object):
  def __init__(self, grid_params, params):
    assert isinstance(grid_params, GridParams) and isinstance(params, TrackerParams)
    self.gp, self.params = grid_params, params

    # self.flow_norm_cost = FlowNormCost(self.u_x_vars, self.u_y_vars)
    # self.opt.add_cost(self.flow_norm_cost, self.params.flow_norm_coeff)

    # self.flow_rigidity_cost = FlowRigidityCost(self.u_x_vars, self.u_y_vars)
    # self.opt.add_cost(self.flow_rigidity_cost, self.params.flow_rigidity_coeff)

    # self.observation_cost = ObservationCost(self.phi_vars)
    # self.opt.add_cost(self.observation_cost, self.params.observation_coeff)

    # self.agreement_cost = AgreementCost(self.phi_vars, self.u_x_vars, self.u_y_vars)
    # self.opt.add_cost(self.agreement_cost, self.params.agreement_coeff)

    # self.prev_weights = None

  def set_prev_phi_and_weights(self, prev_phi, weight):
    self.prev_phi, self.prev_weights = prev_phi, weight

  def set_observation(self, obs, weight):
    self.obs, self.obs_weight = obs, weight

  @staticmethod
  def _solve_quadratic_problem(num_iters, h, phi_init, u_init, v_init, z, w_z, mu_0, mu_u, mu_v, wtilde, alpha, beta, gamma, phi_0, u_0, v_0):
    '''Solves linear systems that arise from Levenberg-Marquardt steps'''

    import scipy.weave
    code = r'''
    typedef blitz::Array<double, 2> Array;

    for (int iter = 0; iter < num_iters; ++iter) {
      Array& phi = (iter % 2 == 0) ? phi_input : mirror_phi_input;
      Array& u = (iter % 2 == 0) ? u_input : mirror_u_input;
      Array& v = (iter % 2 == 0) ? v_input : mirror_v_input;

      Array& mirror_phi = (iter % 2 == 1) ? phi_input : mirror_phi_input;
      Array& mirror_u = (iter % 2 == 1) ? u_input : mirror_u_input;
      Array& mirror_v = (iter % 2 == 1) ? v_input : mirror_v_input;

      for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
          if (i == 0) {

            if (j == 0) {

              mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

              mirror_u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) - 4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) + 2*alpha*v(i+1,j+1) - 2*alpha*v(i+1,j) - 2*alpha*v(i,j+1) + 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

              mirror_v(i,j) = (2*alpha*u(i+1,j+1) - 2*alpha*u(i+1,j) - 2*alpha*u(i,j+1) + 2*alpha*u(i,j) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

            } else if (j == grid_size-1) {

              mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

              mirror_u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) - 4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) - 2*alpha*v(i+1,j-1) + 2*alpha*v(i+1,j) + 2*alpha*v(i,j-1) - 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

              mirror_v(i,j) = (-2*alpha*u(i+1,j-1) + 2*alpha*u(i+1,j) + 2*alpha*u(i,j-1) - 2*alpha*u(i,j) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

            } else {

              mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

              mirror_u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) + 2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) + alpha*v(i+1,j+1) - alpha*v(i+1,j-1) - alpha*v(i,j+1) + alpha*v(i,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_u(i,j)*mu_u(i,j))*wtilde(i,j)));

              mirror_v(i,j) = (alpha*u(i+1,j+1) - alpha*u(i+1,j-1) - alpha*u(i,j+1) + alpha*u(i,j-1) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

            }

          } else if (i == grid_size-1) {

            if (j == 0) {

              mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

              mirror_u(i,j) = (-4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) + 2*alpha*v(i,j+1) - 2*alpha*v(i,j) - 2*alpha*v(i-1,j+1) + 2*alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

              mirror_v(i,j) = (2*alpha*u(i,j+1) - 2*alpha*u(i,j) - 2*alpha*u(i-1,j+1) + 2*alpha*u(i-1,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

            } else if (j == grid_size-1) {

              mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

              mirror_u(i,j) = (-4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) - 2*alpha*v(i,j-1) + 2*alpha*v(i,j) + 2*alpha*v(i-1,j-1) - 2*alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

              mirror_v(i,j) = (-2*alpha*u(i,j-1) + 2*alpha*u(i,j) + 2*alpha*u(i-1,j-1) - 2*alpha*u(i-1,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

            } else {

              mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

              mirror_u(i,j) = (2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) + alpha*v(i,j+1) - alpha*v(i,j-1) - alpha*v(i-1,j+1) + alpha*v(i-1,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_u(i,j)*mu_u(i,j))*wtilde(i,j)));

              mirror_v(i,j) = (alpha*u(i,j+1) - alpha*u(i,j-1) - alpha*u(i-1,j+1) + alpha*u(i-1,j-1) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

            }

          } else {

            if (j == 0) {

              mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

              mirror_u(i,j) = (4*alpha*u(i+1,j) - 4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) + 4*alpha*u(i-1,j) + alpha*v(i+1,j+1) - alpha*v(i+1,j) - alpha*v(i-1,j+1) + alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

              mirror_v(i,j) = (alpha*u(i+1,j+1) - alpha*u(i+1,j) - alpha*u(i-1,j+1) + alpha*u(i-1,j) + 2*alpha*v(i+1,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_v(i,j)*mu_v(i,j))*wtilde(i,j)));

            } else if (j == grid_size-1) {

              mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

              mirror_u(i,j) = (4*alpha*u(i+1,j) - 4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) + 4*alpha*u(i-1,j) - alpha*v(i+1,j-1) + alpha*v(i+1,j) + alpha*v(i-1,j-1) - alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

              mirror_v(i,j) = (-alpha*u(i+1,j-1) + alpha*u(i+1,j) + alpha*u(i-1,j-1) - alpha*u(i-1,j) + 2*alpha*v(i+1,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_v(i,j)*mu_v(i,j))*wtilde(i,j)));

            } else {

              mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

              mirror_u(i,j) = (4*alpha*u(i+1,j) + 2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) + 4*alpha*u(i-1,j) + (1.0L/2.0L)*alpha*v(i+1,j+1) - 1.0L/2.0L*alpha*v(i+1,j-1) - 1.0L/2.0L*alpha*v(i-1,j+1) + (1.0L/2.0L)*alpha*v(i-1,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

              mirror_v(i,j) = ((1.0L/2.0L)*alpha*u(i+1,j+1) - 1.0L/2.0L*alpha*u(i+1,j-1) - 1.0L/2.0L*alpha*u(i-1,j+1) + (1.0L/2.0L)*alpha*u(i-1,j-1) + 2*alpha*v(i+1,j) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

            }

          }
        }
      } // end iteration over grid

      // printf("iter %d done\n", iter);
    }
    '''

    assert u_init.shape[0] == u_init.shape[1]
    assert u_init.shape == v_init.shape == phi_init.shape == z.shape == w_z.shape == mu_0.shape == mu_u.shape == mu_v.shape == wtilde.shape == phi_0.shape == u_0.shape == v_0.shape

    grid_size = u_init.shape[0]

    phi = phi_init.astype(float).copy()
    u = u_init.astype(float).copy()
    v = v_init.astype(float).copy()

    mirror_phi, mirror_u, mirror_v = phi.copy(), u.copy(), v.copy()

    local_dict = {
      'phi_input': phi,
      'u_input': u,
      'v_input': v,
      'mirror_phi_input': mirror_phi,
      'mirror_u_input': mirror_u,
      'mirror_v_input': mirror_v,

      'z': z,
      'w_z': w_z,
      'mu_0': mu_0,
      'mu_u': mu_u,
      'mu_v': mu_v,
      'wtilde': wtilde,
      'alpha': alpha,
      'beta': beta,
      'gamma': gamma,
      'phi_0': phi_0,
      'u_0': u_0,
      'v_0': v_0,

      'h': h,
      'grid_size': grid_size,
      'num_iters': num_iters
    }

    scipy.weave.inline(
      code,
      local_dict.keys(),
      local_dict,
      type_converters=scipy.weave.converters.blitz
    )

    return (phi, u, v) if (num_iters % 2 == 0) else (mirror_phi, mirror_u, mirror_v)

  def _optimize_once(self, init_state, flowed_prev_weights):
    '''Levenberg-Marquardt for the fixed-weight problems'''

    assert self.params.agreement_coeff == 1 and self.params.observation_coeff == 1

    info = {
      'n_qp_solves': 0,
      'n_func_evals': 0,
      'n_iters': 0
    }
    status = 'incomplete'

    exit = False
    curr_x = start_x
    curr_cost, curr_cost_detail = self._eval_true_objective(start_x)
    curr_iter = 0
    trust_region_size = self.init_trust_region_size
    costs_over_iters, x_over_iters = [], []
    while True:
      curr_iter += 1
      costs_over_iters.append(curr_cost)
      x_over_iters.append(curr_x)
      self.logger.info('Starting SQP iteration %d' % curr_iter)

      while trust_region_size >= self.min_trust_region_size:
        self._set_trust_region(trust_region_size, curr_x)
        self.logger.debug('Solving QP')
        self.timer.start('solve_qp')

        phi, u, v = self._solve_quadratic_problem(
          self.params.lin_solver_iters,
          1.,
          init_state.phi, init_state.u_x, init_state.u_y,
          self.obs, self.obs_weight,
          mu_0, mu_u, mu_v,#########################################TODO
          flowed_prev_weights,
          self.params.flow_rigidity_coeff, self.params.flow_norm_coeff,
          gamma, phi_0, u_0, v_0#########################################TODO
        )

        self.timer.end('solve_qp')
        info['n_qp_solves'] += 1

        self.logger.debug('Extracting model costs')
        #model_cost = ???
        self.logger.debug('Evaluating true objective')
        #new_cost = ???

        approx_merit_improve = curr_cost - model_cost
        exact_merit_improve = curr_cost - new_cost

        if approx_merit_improve < -1e-5:
          self.logger.warn("approximate merit function got worse (%.3e). (convexification is probably wrong to zeroth order)" % approx_merit_improve)

        if approx_merit_improve < self.min_approx_improve:
          self.logger.info("converged because improvement was small (%.3e < %.3e)" % (approx_merit_improve, self.min_approx_improve))
          status = 'converged'
          exit = True
          break

        merit_improve_ratio = exact_merit_improve / approx_merit_improve
        if exact_merit_improve < 0 or merit_improve_ratio < self.improve_ratio_threshold:
          trust_region_size *= self.trust_shrink_ratio
          self.logger.info("shrunk trust region. new box size: %.4f" % trust_region_size)
        else:
          curr_x, curr_cost, curr_cost_detail = new_x, new_cost, new_cost_detail
          trust_region_size *= self.trust_expand_ratio
          self.logger.info("expanded trust region. new box size: %.4f" % trust_region_size)
          break

      if exit:
        break

      if trust_region_size < self.min_trust_region_size:
        self.logger.info("converged because trust region is tiny")
        status = 'converged'
        exit = True
        break

      if curr_iter >= self.max_iter:
        self.logger.warn("iteration limit")
        status = 'iter_limit'
        exit = True
        break

    assert exit
    info['n_iters'] = curr_iter
    costs_over_iters.append(curr_cost)
    x_over_iters.append(curr_x)















  def optimize(self, init_phi, init_u_x, init_u_y):
    assert self.params.reweighting_iters >= 1

    init_state = State(self.gp, init_phi, init_u_x, init_u_y)

    if self.params.reweighting_iters == 1:
      return _optimize_once(init_state, self.prev_weights)

    # IRLS loop: Set weights (according to u),
    # solve fixed-weight problem starting from previous solution as initialization, repeat
    curr_state, results, opt_results = init_state, [], []
    flowed_prev_weights = self.prev_weights
    for i in range(self.params.reweighting_iters):
      # solve fixed-weight subproblem
      state, opt_result = self._optimize_once(curr_state, flowed_prev_weights)
      # recalculate weights
      flowed_prev_weights = apply_flow_to_weights(self.gp, self.prev_weights, state.u_x, state.u_y)

      results.append(state)
      opt_results.append(opt_result)
      curr_state = state

    print results[-1].u_x
    return results[-1], opt_results[-1]


def run_one_step(grid_params, tracker_params, obs_tsdf, obs_weight, obs_always_trust_mask, init_phi, init_weight, return_full=False):
  assert isinstance(tracker_params, TrackerParams)

  if return_full:
    problem_data = {}
    problem_data['obs_tsdf'] = obs_tsdf
    problem_data['obs_weight'] = obs_weight
    problem_data['init_phi'] = init_phi
    problem_data['init_weight'] = init_weight

  # Set up tracking problem
  tracker = TrackingOptimizationProblem(grid_params, tracker_params)
  tracker.opt.params().check_linearizations = False
  tracker.opt.params().keep_results_over_iterations = False
  tracker.opt.params().max_iter = tracker_params.max_inner_iters
  tracker.opt.params().approx_improve_rel_tol = 1e-8

  tracker.set_observation(obs_tsdf, obs_weight)
  tracker.set_prev_phi_and_weights(init_phi, init_weight)

  # Run optimization
  # initialization: previous phi, zero flow
  init_u = np.zeros(init_phi.shape + (2,))

  import time
  t_start = time.time()
  result, opt_result = tracker.optimize(init_phi, init_u[:,:,0], init_u[:,:,1])
  print '======== Optimization took %f sec ========' % (time.time() - t_start)

  if return_full:
    problem_data['result'], problem_data['opt_result'] = result, opt_result

  # Flow previous weight to get new weight
  flowed_init_weight = apply_flow_to_weights(grid_params, init_weight, result.u_x, result.u_y)
  new_weight = flowed_init_weight + obs_weight
  if return_full:
    problem_data['new_weight'] = new_weight

  # Smooth next phi
  if tracker_params.enable_smoothing:
    # First smooth by TPS cost to introduce a zero crossing at a good place
    flowed_always_trust_mask = apply_flow(grid_params, np.where(obs_always_trust_mask, 1., 0.), result.u_x, result.u_y) > .1
    smoother_fixed_region = threshold_trusted(tracker_params, result.phi, new_weight, flowed_always_trust_mask)

    # HACK: also fix the far side of the grid to be +trunc
    tmp_phi, tmp_weight = result.phi.copy(), new_weight.copy()
    smoother_fixed_region[:,-1] = True
    tmp_weight[:,-1] = 1
    tmp_phi[:,-1] = tracker_params.tsdf_trunc_dist
    smoother_fixed_region[0,:] = True
    tmp_weight[0,:] = 1
    tmp_phi[0,:] = tracker_params.tsdf_trunc_dist
    smoother_fixed_region[-1,:] = True
    tmp_weight[-1,:] = 1
    tmp_phi[-1,:] = tracker_params.tsdf_trunc_dist

    smoother_weights = np.where(smoother_fixed_region, tmp_weight, 0.)
    new_phi = smooth(tmp_phi, smoother_weights)

    if tracker_params.smoother_post_fmm:
      # March from zero crossing
      new_phi = smooth_to_sdf(new_phi)

  else:
    new_phi = result.phi

  # Re-truncate next phi
  new_phi = np.clip(new_phi, -tracker_params.tsdf_trunc_dist, tracker_params.tsdf_trunc_dist)
  if return_full:
    problem_data['new_phi_smoothed'] = new_phi

  if return_full:
    return new_phi, new_weight, problem_data
  return new_phi, new_weight


def smooth_to_sdf(phi):
  import skfmm
  return skfmm.distance(phi)


def smooth(phi, weights, mode='tps'):
  # from timb_skfmm import distance
  # d = distance(phi, ignore_mask=ignore_mask, order=1)
  # return np.clip(d, -observation.TRUNC_DIST, observation.TRUNC_DIST)

  assert phi.shape[0] == phi.shape[1]
  assert mode in ['laplacian', 'gradient', 'tps']

  gp = GridParams(-1, 1, -1, 1, phi.shape[0], phi.shape[1])
  opt = Optimizer()
  phi_vars = make_var_field(opt, 'phi', gp)

  obs_cost = ObservationCost(phi_vars)
  obs_cost.set_observation(phi, weights)
  opt.add_cost(obs_cost, 1e5)

  if mode == 'laplacian':
    opt.add_cost(LaplacianCost(phi_vars), 1e-10)
  elif mode == 'gradient':
    opt.add_cost(GradientCost(phi_vars), 1e-10)
  elif mode == 'tps':
    opt.add_cost(TPSCost(phi_vars), 1e-10)
  else:
    raise NotImplementedError

  result = opt.optimize(phi.ravel())
  new_phi = result['x'].reshape(phi.shape)
  return new_phi


def threshold_trusted(tracker_params, phi, weight, always_trust_mask=None):
  # The smoother should feel free to overwrite regions of low weight
  # and regions far from zero (unless it's in obs_smoother_ignore_region)
  mask  = abs(phi) < tracker_params.smoother_phi_ignore_thresh
  mask &= weight >= tracker_params.smoother_weight_ignore_thresh
  if always_trust_mask is not None:
    mask |= always_trust_mask
  return mask

def threshold_trusted_for_view(tracker_params, phi, weight):
  return threshold_trusted_for_view2(weight)

def threshold_trusted_for_view2(weight):
  return weight >= .5

def sdf_to_zc(f):
  p = np.pad(f, (1,1), 'edge')
  return (f*p[:-2,1:-1] < 0) | (f*p[2:,1:-1] < 0) | (f*p[1:-1,:-2] < 0) | (f*p[1:-1,2:] < 0)

########## Utility functions ##########

def plot_state(state):
  import matplotlib
  import matplotlib.pyplot as plt
  plt.clf()
  matplotlib.rcParams.update({'font.size': 8})

  TSDF_TRUNC = 3.
  plt.subplot(121)
  plt.title('phi')
  plt.axis('off')
  # FIXME: this is wrong
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


def plot_problem_data(plt, tsdf_trunc_dist, gp, state, tsdf, obs_weight, init_phi, init_weight, result, opt_result, next_phi, next_weight, output):

  def plot_field(f, contour=False):
    plt.imshow(f.T, aspect=1, vmin=-tsdf_trunc_dist, vmax=tsdf_trunc_dist, cmap='bwr', origin='lower')
    if contour:
      x = np.linspace(gp.xmin, gp.xmax, gp.nx)
      y = np.linspace(gp.ymin, gp.ymax, gp.ny)
      X, Y = np.meshgrid(x, y, indexing='ij')
      plt.contour(X, Y, f, levels=[0])

  def plot_u(u_x, u_y):
    assert u_x.shape == u_y.shape
    x = np.linspace(gp.xmin, gp.xmax, gp.nx)
    y = np.linspace(gp.ymin, gp.ymax, gp.ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    from scipy.ndimage.interpolation import zoom
    a = .3
    plt.quiver(zoom(X, a), zoom(Y, a), zoom(u_x, a), zoom(u_y, a), angles='xy', scale_units='xy', scale=1.)

  plt.clf()
  import matplotlib
  matplotlib.rcParams.update({'font.size': 8, 'image.origin': 'lower'})

  plt.subplot(251)
  plt.title('State')
  plt.axis('off')
  plt.imshow(state.T, aspect=1, origin='lower')

  plt.subplot(252)
  plt.axis('off')
  plt.title('Observation TSDF')
  plot_field(tsdf, contour=True)

  plt.subplot(253)
  plt.title('Observation weight')
  plt.axis('off')
  plt.imshow(obs_weight.T, cmap='binary', vmin=0, vmax=observation.OBS_PEAK_WEIGHT*10).set_interpolation('nearest')

  plt.subplot(254)
  plt.title('Prior TSDF')
  plt.axis('off')
  plot_field(init_phi, contour=True)

  plt.subplot(255)
  plt.title('Prior weight')
  plt.axis('off')
  plt.imshow(init_weight.T, vmin=0, vmax=observation.OBS_PEAK_WEIGHT*10, cmap='binary').set_interpolation('nearest')

  # plt.subplot(256)
  # plt.title('Log cost')
  # plt.plot(np.log(opt_result['cost_over_iters']))

  plt.subplot(256, aspect='equal')
  plt.title('Flow')
  plt.axis('off')
  plot_u(result.u_x, result.u_y)

  plt.subplot(257)
  plt.title('New TSDF')
  plt.axis('off')
  plot_field(result.phi, contour=True)

  plt.subplot(258)
  plt.title('New weight')
  plt.axis('off')
  plt.imshow(next_weight.T, vmin=0, vmax=observation.OBS_PEAK_WEIGHT*10, cmap='binary').set_interpolation('nearest')

  plt.subplot(259)
  plt.title('Smoothed new TSDF')
  plt.axis('off')
  plot_field(next_phi, contour=True)

  plt.subplot(2, 5, 10)
  plt.title('Output')
  plt.axis('off')
  plot_field(output, contour=True)
