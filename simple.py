import numpy as np
import sqp_problem

SIZE = 4
WORLD_MIN = (0., 0.)
WORLD_MAX = (3., 3.)
PIXEL_AREA = 4./SIZE/SIZE
PIXEL_SIDE = 2./SIZE

sqp_problem.GRID_NX = SIZE
sqp_problem.GRID_NY = SIZE
sqp_problem.GRID_SHAPE = (SIZE, SIZE)
sqp_problem.GRID_MIN = WORLD_MIN
sqp_problem.GRID_MAX = WORLD_MAX
sqp_problem.PIXEL_AREA = PIXEL_AREA

def plot_phi(phi):
  n = 256
  x = np.linspace(WORLD_MIN[0], WORLD_MAX[0], n)
  y = np.linspace(WORLD_MIN[1], WORLD_MAX[1], n)
  X, Y = np.meshgrid(x, y)
  Z = phi.eval_xys(np.dstack((X, Y)).reshape((-1, 2))).reshape((n, n))

  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(X, Y, Z, cmap='hot')
  plt.show()

def main():
  empty_phi = np.ones((4, 4))
  init_phi = np.array([
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
  ])
  obs = np.array([
    [1, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
  ], dtype=bool)
  obs_pts = np.transpose(np.nonzero(obs))

  prob = sqp_problem.TrackingProblem()
  prob.set_obs_points(obs_pts)
  prob.set_prev_phi_surf(sqp_problem.make_bicubic(init_phi))

  prob.set_coeffs(flow_norm=0, flow_rigidity=10, obs=1, flow_agree=1)

  init_u = np.zeros((4, 4, 2))
  out_phi_surf, out_u = prob.optimize(init_u)
  plot_phi(out_phi_surf)
  import IPython; IPython.embed()

if __name__ == '__main__':
  main()
