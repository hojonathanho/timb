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

  def set_observation(self, obs, weights):
    self.observation_cost.set_observation(obs, weights)

  def optimize(self, init_state):
    def _optimize_once(state):
      opt_result = self.opt.optimize(state.pack())
      result = State.FromPacked(self.gp, opt_result['x'])
      return result, opt_result

    assert self.params.reweighting_iters >= 1
    if self.params.reweighting_iters == 1:
      return _optimize_once(init_state)

    curr_state, results, opt_results = init_state, [], []
    for i in range(self.params.reweighting_iters):
      state, opt_result = _optimize_once(curr_state)

      flowed_prev_weights = apply_flow_to_weights(self.gp, self.prev_weights, state.u_x, state.u_y)
      self.set_prev_phi_and_weights(self.prev_phi, flowed_prev_weights) # prev_phi stays the same, only weights change

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
  init_state = State(grid_params, init_phi, init_u[:,:,0], init_u[:,:,1])
  result, opt_result = tracker.optimize(init_state)
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
