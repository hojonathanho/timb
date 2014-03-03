import ctimb
from ctimb import *
import numpy as np
from observation import * 

np.set_printoptions(linewidth=1000)

class RigidTrackerParams(object):
  def __init__(self):
    # Optimization parameters
    self.max_iters = 50
    self.check_linearizations = False
    self.keep_results_over_iteration = False
    self.approx_improve_rel_tol = 1e-10

    # Observation parameters
    self.tsdf_trunc_dist = 20.
    self.sensor_mode = 'accurate'

    self.obs_weight_epsilon = 0
    self.obs_weight_delta = 5.
    self.obs_weight_filter_radius = 10
    self.obs_weight_far = True

    self.use_linear_downweight = True
    self.use_min_to_combine = True


def optimize_sdf_transform(phi, weight, grid_params, obs_zero_points, tracker_params, init_x=0, init_y=0, init_theta=0.0):
  """
  optimizes for the movement of the SDF.
  phi         : old TSDF (TSDF at previous time step)
  grid_params : grid parameters of the TSDF
  obs         : new obseravations (2D matrix of (x_i, y_i))
  init_x/y/theta : initial guess for the change in SDF pose.
  """
  assert isinstance(grid_params, GridParams)

  opt = Optimizer()
  opt.params().check_linearizations         = tracker_params.check_linearizations
  opt.params().keep_results_over_iterations = tracker_params.keep_results_over_iteration
  opt.params().max_iter = tracker_params.max_iters
  opt.params().approx_improve_rel_tol = tracker_params.approx_improve_rel_tol
  
  tsdf = make_double_field(grid_params, phi)
  wfield = make_double_field(grid_params, weight)
  dx_var, dy_var, dth_var = opt.add_vars(["dx", "dy", "dth"])
  obs_zc_cost = RigidObservationZeroCrossingCost(tsdf, wfield, dx_var, dy_var, dth_var)
  obs_zc_cost.set_zero_points(obs_zero_points)
  opt.add_cost(obs_zc_cost)

  ## cost for norm of dx,dy,dth
  disp_cost = DisplacementCost(dx_var, dy_var, dth_var)
  opt.add_cost(disp_cost)

  opt_result = opt.optimize(np.array([init_x, init_y, init_theta]))
  return opt_result
  

def transform_last_tsdf(old_phi, old_weight, grid_params, dx,dy,dth):
  old_tsdf   = make_double_field(grid_params, old_phi)
  new_phi    = apply_rigid_transform(old_tsdf, dx,dy,dth)

  old_w_field  = make_double_field(grid_params, old_weight)
  new_weight   = apply_rigid_transform(old_w_field, dx,dy,dth)
  
  return new_phi, new_weight


def average_tsdfs(phi0, w0, phi1, w1):
  phi_n = (phi0*w0 + phi1*w1) / (w0 + w1 + np.spacing(0))
  w_n   = w0 + w1
  return phi_n, w_n


def grid_ij_to_xy(i,j, grid_params):
  return (grid_params.xmin + i*grid_params.eps_x, 
          grid_params.ymin + j*grid_params.eps_y);


def run_one_rigid_step(grid_params, tracker_params, obs_depth, obs_tsdf, obs_weight, prev_tsdf, prev_weight, return_full=False):
    assert isinstance(tracker_params, RigidTrackerParams)

    if return_full:
      problem_data = {}
      problem_data['obs_depth']   = obs_depth
      problem_data['obs_weight']  = obs_weight
      problem_data['prev_tsdf']   = prev_tsdf
      problem_data['prev_weight'] = prev_weight
      problem_data['result']      = None
      problem_data['opt_result']  = None

    obs_ij = np.c_[np.arange(len(obs_depth)), obs_depth]
    obs_xy = np.empty(obs_ij.shape)
    for r in xrange(len(obs_xy)):
      obs_xy[r] = grid_ij_to_xy(obs_ij[r,0], obs_ij[r,1], grid_params)

    opt_result = optimize_sdf_transform(prev_tsdf, prev_weight, grid_params, obs_xy, tracker_params)
    if return_full:
      problem_data['opt_result'] = opt_result
    [dx, dy, dth] =  opt_result['x']

    tf_tsdf, tf_weight  = transform_last_tsdf(prev_tsdf, prev_weight, grid_params, dx,dy,dth)
    new_phi, new_weight = average_tsdfs(tf_tsdf, tf_weight, obs_tsdf, obs_weight)
    
    if return_full:
      return new_phi, new_weight, obs_xy, problem_data
    else:
      return new_phi, new_weight


def plot_problem_data(plt, tsdf_trunc_dist, gp, state, obs_xy, obs_tsdf, obs_weight, prev_tsdf, prev_weight, new_tsdf, new_weight, out_state):
  
  def plot_field(f, contour=False):
    plt.imshow(f.T, aspect=1, vmin=-tsdf_trunc_dist, vmax=tsdf_trunc_dist, cmap='bwr', origin='lower')
    if contour:
      x = np.linspace(gp.xmin, gp.xmax, gp.nx)
      y = np.linspace(gp.ymin, gp.ymax, gp.ny)
      X, Y = np.meshgrid(x, y, indexing='ij')
      plt.contour(X, Y, f, levels=[0])

  plt.clf()
  import matplotlib
  matplotlib.rcParams.update({'font.size': 8, 'image.origin': 'lower'})

  plt.subplot(251)
  plt.imshow(state.T, aspect=1, origin='lower')
  plt.title('state')

  plt.subplot(252)
  plot_field(obs_tsdf, contour=True)
  #plt.imshow(obs_tsdf, cmap='bwr', vmin=-tsdf_trunc_dist, vmax=tsdf_trunc_dist).set_interpolation('nearest')
  #plt.contour(obs_tsdf, levels=[0])
  plt.plot(obs_xy[:,0], obs_xy[:,1])
  plt.title('observation tsdf')

  plt.subplot(253)
  plt.imshow(obs_weight.T, cmap='binary', vmin=0, vmax=10).set_interpolation('nearest')
  plt.title('observation weight')

  plt.subplot(254)
  plot_field(prev_tsdf, contour=True)
  plt.title('prior tsdf')

  plt.subplot(255)
  plt.imshow(prev_weight.T, cmap='binary', vmin=0, vmax=10).set_interpolation('nearest')
  plt.title('prior weight')
    
  plt.subplot(257)
  plot_field(new_tsdf, contour=True)
  plt.title('new tsdf')

  plt.subplot(258)
  plt.imshow(new_weight.T, cmap='binary', vmin=0, vmax=10).set_interpolation('nearest')
  plt.title('new weight')
  
  plt.subplot(259)
  plot_field(out_state, contour=True)
  plt.title('out state')


def threshold_trusted_for_view(weight):
  return weight >= .5


def test_one_step():
  TSDF_TRUNC = 20
  SIZE = 100
  WORLD_MIN = (0., 0)
  WORLD_MAX = (SIZE-1, SIZE-1)
  
  gp = ctimb.GridParams(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], SIZE, SIZE)
  tracker_params =  RigidTrackerParams()
  tracker_params.tsdf_trunc_dist = TSDF_TRUNC
  
  ## define the states at time t=0 and t=1:
  state0 = np.zeros((SIZE,SIZE), dtype=bool)
  state0[10:60, 10:60] = np.ones((50,50), dtype=bool)
  w0 = np.zeros((SIZE, SIZE)); w0.fill(1.)

  state1 = np.zeros((SIZE,SIZE), dtype=bool)
  state1[15:65, 15:65] = np.ones((50,50), dtype=bool)
  
  ## generate observations:
  tsdf0, sdf0, _, _, _ = observation_from_full_state(state0, tracker_params)
  w0 = np.zeros_like(state0)
  tsdf1, sdf1, depth1, w1, _ = observation_from_full_state(state1, tracker_params)

  ## optimize for camera pose and find the new sdf:
  tsdf1_opt, w1_opt = run_one_rigid_step(gp, tracker_params, depth1, tsdf1, w1, tsdf0, w0, return_full=False)

  out_state = np.where(threshold_trusted_for_view(w1_opt), tsdf1_opt, np.nan)

  #import matplotlib.pyplot as plt
  #plot_problem_data(plt, state1, tsdf1, w1, tsdf0, w0, tsdf1_opt, w1_opt, out_state) 
  #plt.show()


if __name__=="__main__":
  test_one_step()
