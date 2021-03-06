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
import logging


class TrackerParams(object):
  def __init__(self):
    # Optimization parameters
    self.flow_norm_coeff = 1e-6
    self.flow_rigidity_coeff = 1.
    self.observation_coeff = 1.
    self.agreement_coeff = 1.

    self.reweighting_iters = 10 # iterative reweighted least squares iterations
    self.max_inner_iters = 10 # Levenberg-Marquardt iterations
    self.lin_solver_iters = 30 # iterations for iterative linear system solver

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
  '''Tracking optimization problem class. Uses the factorization-based solver in src/optimizer.cpp'''

  def __init__(self, grid_params, params):
    assert isinstance(grid_params, GridParams) and isinstance(params, TrackerParams)
    self.gp, self.params = grid_params, params

    self.opt = Optimizer()
    self.opt.params().check_linearizations = False
    self.opt.params().keep_results_over_iterations = False
    self.opt.params().max_iter = self.params.max_inner_iters
    self.opt.params().approx_improve_rel_tol = 1e-8

    def cb(old_x, delta_x, true_old_cost, true_improvement, model_improvement, ratio):
      print '\n===================CALLBACK==================='
      print old_x, delta_x, true_old_cost, true_improvement, model_improvement, ratio
      old_state = State.FromPacked(self.gp, old_x)
      print 'Old cost:', true_old_cost
      print 'Old cost from hardcoded:', timb_problem_eval_objective(
        self.gp,
        old_state.phi, old_state.u_x, old_state.u_y,
        self.obs, self.obs_weights,
        self.prev_phi, self.curr_wtilde,
        self.params.flow_rigidity_coeff, self.params.flow_norm_coeff
      )
      print 'Model improvement:', model_improvement
      print 'Model improvement from hardcoded:'
      mu_0, mu_u, mu_v = timb_linearize_flowed_prev_phi(self.gp, self.prev_phi, old_state.u_x, old_state.u_y)
      new_state = State.FromPacked(self.gp, old_x + delta_x)
      print true_old_cost - timb_problem_eval_model_objective(
        self.gp,
        new_state.phi, new_state.u_x, new_state.u_y,
        self.obs, self.obs_weights,
        mu_0, mu_u, mu_v, self.curr_wtilde,
        self.params.flow_rigidity_coeff, self.params.flow_norm_coeff
      )
      print '===================END CALLBACK===================\n'
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

    # Set up logging
    self.logger = logging.getLogger('LM')
    self.logger.setLevel(logging.INFO)
    if not self.logger.handlers:
      ch = logging.StreamHandler()
      # ch.setLevel(logging.DEBUG)
      ch.setFormatter(logging.Formatter('>>> %(name)s - %(levelname)s - %(message)s'))
      self.logger.addHandler(ch)

  def set_prev_phi_and_weights(self, prev_phi, weight):
    self.prev_phi, self.prev_weights = prev_phi, weight

  def set_observation(self, obs, weight):
    self.obs, self.obs_weight = obs, weight

  def _optimize_once(self, start_state, flowed_prev_weights, max_levmar_iters):
    '''Levenberg-Marquardt for the fixed-weight problems'''

    assert self.params.agreement_coeff == 1 and self.params.observation_coeff == 1

    def eval_true_objective(state):
      return timb_problem_eval_objective(
        self.gp,
        state.phi, state.u_x, state.u_y,
        self.obs, self.obs_weight,
        self.prev_phi, flowed_prev_weights,
        self.params.flow_rigidity_coeff, self.params.flow_norm_coeff
      )

    def compute_model_data(state):
      # compute data for quadratic model of objective around the given values of u, v (in state)
      # i.e. linearize the flowed prev_phi w.r.t. u, v
      return timb_linearize_flowed_prev_phi(self.gp, self.prev_phi, state.u_x, state.u_y)

    def eval_model_objective(state, model_data):
      '''model_data is a tuple (mu_0, mu_u, mu_v)'''
      return timb_problem_eval_model_objective(
        self.gp,
        state.phi, state.u_x, state.u_y,
        self.obs, self.obs_weight,
        model_data[0], model_data[1], model_data[2], flowed_prev_weights,
        self.params.flow_rigidity_coeff, self.params.flow_norm_coeff
      )

    def solve_model_problem(init_state, model_data, trust_region_data):
      print 'DAMPING:', trust_region_data[0]
      '''trust_region_data is a tuple (damping, center for phi, center for u, center for v)'''
      out_phi, out_u, out_v = timb_solve_model_problem_gauss_seidel(
        self.gp,
        init_state.phi, init_state.u_x, init_state.u_y,
        self.obs, self.obs_weight,
        model_data[0], model_data[1], model_data[2], flowed_prev_weights,
        self.params.flow_rigidity_coeff, self.params.flow_norm_coeff,
        trust_region_data[0], trust_region_data[1], trust_region_data[2], trust_region_data[3],
        self.params.lin_solver_iters
      )
      return State(self.gp, out_phi, out_u, out_v)

    info = {
      'n_qp_solves': 0,
      'n_func_evals': 0,
      'n_iters': 0
    }
    status = 'incomplete'

    INIT_DAMPING = 1e-5
    MAX_DAMPING = 1e5
    MIN_DAMPING = 1e-10
    DAMPING_DECREASE_RATIO = .5
    DAMPING_INCREASE_RATIO = 10.
    IMPROVE_RATIO_THRESHOLD = .25
    MIN_APPROX_IMPROVE = 1e-6

    exit = False
    curr_state = start_state
    curr_cost = eval_true_objective(start_state)
    curr_iter = 0
    damping = INIT_DAMPING
    costs_over_iters, x_over_iters = [], []
    while True:
      curr_iter += 1
      costs_over_iters.append(curr_cost)
      x_over_iters.append(curr_state)
      self.logger.info('Starting SQP iteration %d' % curr_iter)

      while damping <= MAX_DAMPING:
        self.logger.debug('Solving QP')
        # self.timer.start('solve_qp')
        model_data = compute_model_data(curr_state)
        new_state = solve_model_problem(
          curr_state, model_data,
          (damping, curr_state.phi, curr_state.u_x, curr_state.u_y)
        )
        # self.timer.end('solve_qp')
        info['n_qp_solves'] += 1

        self.logger.debug('Extracting model costs')
        model_cost = eval_model_objective(new_state, model_data)
        self.logger.debug('Evaluating true objective')
        new_cost = eval_true_objective(new_state)

        approx_merit_improve = curr_cost - model_cost
        exact_merit_improve = curr_cost - new_cost

        if approx_merit_improve < -1e-5:
          self.logger.warn("approximate merit function got worse (%.3e). (convexification is probably wrong to zeroth order)" % approx_merit_improve)

        if approx_merit_improve < MIN_APPROX_IMPROVE:
          self.logger.info("converged because improvement was small (%.3e < %.3e)" % (approx_merit_improve, MIN_APPROX_IMPROVE))
          status = 'converged'
          exit = True
          break

        merit_improve_ratio = exact_merit_improve / approx_merit_improve
        if exact_merit_improve < 0 or merit_improve_ratio < IMPROVE_RATIO_THRESHOLD:
          damping *= DAMPING_INCREASE_RATIO
          self.logger.info("increased damping to %.4f" % damping)
        else:
          curr_state, curr_cost = new_state, new_cost
          damping = max(MIN_DAMPING, damping * DAMPING_DECREASE_RATIO)
          self.logger.info("decreased damping to %.4f" % damping)
          break

      if exit:
        break

      if damping > MAX_DAMPING:
        self.logger.info("converged because damping is too large")
        status = 'converged'
        exit = True
        break

      if curr_iter >= max_levmar_iters:
        self.logger.warn("iteration limit")
        status = 'iter_limit'
        exit = True
        break

    assert exit
    info['n_iters'] = curr_iter
    costs_over_iters.append(curr_cost)
    x_over_iters.append(curr_state)

    info['status'] = status
    return curr_state, info


  def optimize(self, init_phi, init_u_x, init_u_y):
    assert self.params.reweighting_iters >= 1

    init_state = State(self.gp, init_phi, init_u_x, init_u_y)

    if self.params.reweighting_iters == 1:
      return self._optimize_once(init_state, self.prev_weights, self.params.max_inner_iters)

    # IRLS loop: Set weights (according to u),
    # solve fixed-weight problem starting from previous solution as initialization, repeat
    curr_state, results, opt_results = init_state, [], []
    flowed_prev_weights = self.prev_weights
    for i in range(self.params.reweighting_iters):
      # solve fixed-weight subproblem
      state, opt_result = self._optimize_once(curr_state, flowed_prev_weights, self.params.max_inner_iters)
      # recalculate weights
      flowed_prev_weights = apply_flow_to_weights(self.gp, self.prev_weights, state.u_x, state.u_y)

      results.append(state)
      opt_results.append(opt_result)
      curr_state = state

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
  tracker = TrackingOptimizationProblem2(grid_params, tracker_params)
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
