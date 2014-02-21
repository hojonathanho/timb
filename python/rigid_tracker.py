import ctimb
from ctimb import *
import numpy as np
from observation import * 
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=1000)

class RigidTrackerParams(object):
  def __init__(self):
    # Optimization parameters
    self.max_iters = 50
    self.check_linearizations = False
    self.keep_results_over_iteration = False
    self.approx_improve_rel_tol = 1e-10

    # Observation parameters
    self.tsdf_trunc_dist = 10.
    self.sensor_mode = 'accurate'

    self.obs_weight_epsilon = 0.
    self.obs_weight_delta = 5.
    self.obs_weight_filter_radius = 5
    self.obs_weight_far = True

    self.use_linear_downweight = True
    self.use_min_to_combine = True


def optimize_sdf_transform(phi, grid_params, obs_zero_points, tracker_params, init_x=0, init_y=0, init_theta=0):
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
  dx_var, dy_var, dth_var = opt.add_vars(["dx", "dy", "dth"])
  obs_zc_cost = RigidObservationZeroCrossingCost(tsdf, dx_var, dy_var, dth_var)
  obs_zc_cost.set_zero_points(obs_zero_points)
  opt.add_cost(obs_zc_cost)

  opt_result = opt.optimize(np.array([init_x, init_y, init_theta]))
  return opt_result
  

def update_last_tsdf(old_phi, grid_params, dx,dy,dth):
  old_tsdf   = make_double_field(grid_params, old_phi)
  new_phi    = apply_rigid_transform(old_tsdf, dx,dy,dth)
  return new_phi


def average_tsdfs(phi0, w0, phi1, w1):
  phi_n = phi0*w0 + phi1*w1 / (w0 + w1)
  w_n   = w0 + w1
  return phi_n, w_n


def grid_ij_to_xy(i,j, grid_params):
  return (grid_params.xmin + i*grid_params.eps_x, 
          grid_params.ymin + j*grid_params.eps_y);


def run_one_rigid_step(grid_params, tracker_params, obs_depth, obs_weight, obs_always_trust_mask, init_phi, init_weight, return_full=False):
    assert isinstance(tracker_params, RigidTrackerParams)

    init_tsdf = np.clip(init_phi, -tracker_params.tsdf_trunc_dist, tracker_params.tsdf_trunc_dist)

    if return_full:
      problem_data = {}
      problem_data['obs_depth']   = obs_depth
      problem_data['obs_weight']  = obs_weight
      problem_data['init_phi']    = init_phi
      problem_data['init_tsdf']   = init_tsdf
      problem_data['init_weight'] = init_weight
      problem_data['result']      = None
      problem_data['opt_result']  = None


    obs_ij = np.c_[np.arange(len(obs_depth)), obs_depth]
    obs_xy = np.empty(obs_ij.shape)
    for r in xrange(len(obs_xy)):
      obs_xy[r] = grid_ij_to_xy(obs_ij[r,0], obs_ij[r,1], grid_params)

    ##TODO : Should I use init_phi/ init_tsdf for camera pose optimization??
    opt_result = optimize_sdf_transform(init_tsdf, grid_params, obs_xy, tracker_params)
    if return_full:
      problem_data['opt_result'] = opt_result
    [dx, dy, dth] = opt_result['x']

    new_tsdf = update_last_tsdf(init_tsdf, grid_params, dx,dy,dth)
    new_phi, new_weight = average_tsdfs(init_tsdf, init_weight, new_tsdf, obs_weight)
    
    if return_full:
      return new_phi, new_weight, problem_data
    else:
      return new_phi, new_weight
    
    

def test_one_step():

  TSDF_TRUNC = 10
  SIZE = 20
  WORLD_MIN = (0., 0)
  WORLD_MAX = (SIZE-1, SIZE-1)
  
  gp = ctimb.GridParams(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], SIZE, SIZE)
  tracker_params =  RigidTrackerParams()
  tracker_params.tsdf_trunc_dist = TSDF_TRUNC
  
  ## define the states at time t=0 and t=1:
  state0 = np.zeros((SIZE,SIZE), dtype=bool)
  state0[10:15, 10:14] = np.ones((5,4), dtype=bool)
  w0 = np.zeros((SIZE, SIZE)); w0.fill(1.)

  state1 = np.zeros((SIZE,SIZE), dtype=bool)
  state1[10:15, 5:9] = np.ones((5,4), dtype=bool)
  
  ## generate observations:
  tsdf0, sdf0, _, _, _ = observation_from_full_state(state0, tracker_params)
  w0 = np.zeros_like(state0)
  tsdf1, sdf1, depth1, w1, _ = observation_from_full_state(state1, tracker_params)
  w1 = np.ones_like(state1)
  
  ## optimize for camera pose and find the new sdf:
  tsdf1_opt, w1_opt = run_one_rigid_step(gp, tracker_params, depth1, w1, None, sdf0, w0, return_full=False)

  plt.subplot(321)
  plt.imshow(state0)
  plt.title('state 0')
  plt.subplot(322)
  plt.imshow(state1)
  plt.title('state 1')
  plt.subplot(323)
  plt.imshow(tsdf0, cmap='bwr').set_interpolation('nearest')
  plt.contour(tsdf0, levels=[0])
  plt.title('tsdf 0')
  plt.subplot(324)
  plt.imshow(tsdf1_opt, cmap='bwr').set_interpolation('nearest')
  plt.contour(tsdf1_opt, levels=[0])
  plt.title('tsdf 1 opt')
  plt.subplot(325)
  plt.imshow(tsdf1, cmap='bwr').set_interpolation('nearest')
  plt.contour(tsdf1, levels=[0])
  plt.title('tsdf 1 true')

  plt.show()


if __name__=="__main__":
  test_one_step()