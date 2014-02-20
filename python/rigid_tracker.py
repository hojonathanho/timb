import ctimb
from ctimb import *
import numpy as np
from observation import * 
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=1000)


def optimize_sdf_transform(phi, grid_params, obs_zero_points, init_x=0, init_y=0, init_theta=0):
  """
  optimizes for the movement of the SDF.
  phi         : old TSDF (TSDF at previous time step)
  grid_params : grid parameters of the TSDF
  obs         : new obseravations (2D matrix of (x_i, y_i))
  init_x/y/theta : initial guess for the change in SDF pose.
  """
  assert isinstance(grid_params, GridParams)
  opt = Optimizer()
  opt.params().check_linearizations         = False
  opt.params().keep_results_over_iterations = False
  opt.params().max_iter = 10
  opt.params().approx_improve_rel_tol = 1e-10
  
  tsdf = make_double_field(grid_params, phi)
  dx_var, dy_var, dth_var = opt.add_vars(["dx", "dy", "dth"])
  obs_zc_cost = RigidObservationZeroCrossingCost(tsdf, dx_var, dy_var, dth_var)
  obs_zc_cost.set_zero_points(obs_zero_points)
  opt.add_cost(obs_zc_cost)
  
  opt_result = opt.optimize(np.array([init_x, init_y, init_theta]))
  return opt_result


def grid_to_xy(i,j, grid_params):
  return (grid_params.xmin + i*grid_params.eps_x, 
          grid_params.ymin + j*grid_params.eps_y);


def test_pose_opt():
  
  TSDF_TRUNC = 50
  SIZE = 20
  WORLD_MIN = (0., 0)
  WORLD_MAX = (SIZE-1, SIZE-1)
  gp = ctimb.GridParams(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], SIZE, SIZE)
  print "grid params ", gp

  state0 = np.zeros((SIZE,SIZE), dtype=bool)
  state0[10:15, 10:14] = np.ones((5,4), dtype=bool)
  state1 = np.zeros((SIZE,SIZE), dtype=bool)
  state1[10:15, 5:9] = np.ones((5,4), dtype=bool)

  print state0, state1


  tsdf0, sdf0, depth0 = state_to_tsdf(state0, TSDF_TRUNC, return_all=True)
  tsdf1, sdf1, depth1 = state_to_tsdf(state1, TSDF_TRUNC, return_all=True)
  plt.subplot(121)
  plt.imshow(tsdf0)
  plt.subplot(122)
  plt.imshow(tsdf1)
  plt.show()

  obs_ij = np.c_[np.arange(len(depth1)), depth1]
  obs_xy = np.empty(obs_ij.shape)
  for r in xrange(len(obs_xy)):
    obs_xy[r] = grid_to_xy(obs_ij[r,0], obs_ij[r,1], gp)
    
  print "depths : "
  print obs_xy
  
  print "opt pose : ", optimize_sdf_transform(tsdf0, gp, obs_xy)


if __name__=="__main__":
  test_pose_opt()