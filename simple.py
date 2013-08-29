import numpy as np
import sqp_problem
np.set_printoptions(linewidth=10000)

SIZE = 4
WORLD_MIN = (0., 0.)
WORLD_MAX = (3., 3.)
sqp_problem.Config.set(SIZE, SIZE, WORLD_MIN, WORLD_MAX)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

def plot_phi(phi):
  n = 256
  x = np.linspace(WORLD_MIN[0], WORLD_MAX[0], n)
  y = np.linspace(WORLD_MIN[1], WORLD_MAX[1], n)
  X, Y = np.meshgrid(x, y)
  Z = phi.eval_xys(np.dstack((X, Y)).reshape((-1, 2))).reshape((n, n))


  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(X, Y, Z, cmap='hot')

def plot_u(u):
  ax = fig.add_subplot(211)
  x = np.linspace(WORLD_MIN[0], WORLD_MAX[0], u.shape[0])
  y = np.linspace(WORLD_MIN[1], WORLD_MAX[1], u.shape[1])
  X, Y = np.meshgrid(x, y)
  ax.quiver(X, Y, u[:,:,0], u[:,:,1])

def main():
  empty_phi = np.ones((4, 4))
  init_phi = np.array([
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
  ], dtype=float)
  # init_phi = np.array([
  #   [1, .5, .5, 1],
  #   [.5, 0, 0, .5],
  #   [1, .5, .5, 1],
  #   [1, 1, 1, 1]
  # ], dtype=float)
  # desired_phi_soln = np.array([
  #   [1, 1, 1, 1],
  #   [1, 1, 1, 1],
  #   [1, 0, 0, 1],
  #   [1, 1, 1, 1]
  # ])
  obs = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0]
  ], dtype=bool)
  obs_pts = np.transpose(np.nonzero(obs))

  prob = sqp_problem.TrackingProblem()
  prob.set_obs_points(obs_pts)
  prob.set_prev_phi_surf(sqp_problem.make_interp(init_phi))

  prob.set_coeffs(flow_norm=0, flow_rigidity=10, obs=1, flow_agree=1)

  init_u = np.zeros((4, 4, 2))
  init_u[:,:,0] = 1.
  out_phi_surf, out_u = prob.optimize(init_phi, init_u)
  plot_phi(out_phi_surf)
  # plot_u(out_u)

  print out_phi_surf.data
  print out_u
  plt.show()
  #import IPython; IPython.embed()

if __name__ == '__main__':
  main()
