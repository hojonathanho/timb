import numpy as np
import sqp_problem
import scipy.ndimage as sn
np.set_printoptions(linewidth=10000)

SIZE = 30
WORLD_MIN = (0., 0.)
WORLD_MAX = (SIZE-1., SIZE-1.)
sqp_problem.Config.set(SIZE, SIZE, WORLD_MIN, WORLD_MAX)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

def plot_phi(N, phi):
  x = np.linspace(WORLD_MIN[0], WORLD_MAX[0], phi.nx)
  y = np.linspace(WORLD_MIN[1], WORLD_MAX[1], phi.ny)
  X, Y = np.meshgrid(x, y)
  Z = phi.eval_xys(np.dstack((X, Y)).reshape((-1, 2))).reshape((phi.nx, phi.ny))

  ax = fig.add_subplot(N)
  ax.imshow(Z, cmap=cm.Greys_r).set_interpolation('nearest')

  # ax = fig.add_subplot(N, projection='3d')
  # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

def plot_u(N, u):
  x = np.linspace(WORLD_MIN[0], WORLD_MAX[0], u.shape[0])
  y = np.linspace(WORLD_MIN[1], WORLD_MAX[1], u.shape[1])
  X, Y = np.meshgrid(x, y)
  ax = fig.add_subplot(N)
  ax.quiver(X, Y, u[:,:,1], u[:,:,0])

def smooth(phi):
  d = sn.morphology.distance_transform_edt(phi)
  d /= abs(d).max()
  return d

def main():
  empty_phi = np.ones((SIZE, SIZE))

  square_phi = np.ones_like(empty_phi)
  square_phi[int(SIZE/2.),int(SIZE/4.):-int(SIZE/4.)] = 0.
  square_phi[int(SIZE/2.):-int(SIZE/4.),int(SIZE/4.)] = 0.
  square_phi[int(SIZE/2.):-int(SIZE/4.),-int(SIZE/4.)] = 0.
  square_phi[-int(SIZE/4.),int(SIZE/4.):-int(SIZE/4.)+1] = 0.
  init_phi = square_phi
  # init_phi = np.ones_like(empty_phi)
  # init_phi[int(SIZE/2.),int(SIZE/4.):-int(SIZE/4.)] = 0.
  # init_phi[int(SIZE/4.),int(SIZE/3.):-int(SIZE/3.)] = 0.
  init_phi = smooth(init_phi)

  init_phi_observed = np.ones_like(empty_phi)
  init_phi_observed[int(SIZE/2.),int(SIZE/4.):-int(SIZE/4.)] = 0.
  # init_phi_observed = smooth(init_phi_observed)

  # new_phi = sn.interpolation.shift(init_phi_observed, [0,1], cval=1., order=0)
  new_phi_observed = sn.interpolation.rotate(init_phi_observed, 1., cval=1., order=0)
  # plt.imshow(new_phi_observed, cmap=cm.Greys_r)
  # plt.show()
  obs_pts = np.transpose(np.nonzero(new_phi_observed == 0))

  prob = sqp_problem.TrackingProblem()
  prob.set_obs_points(obs_pts)
  prob.set_prev_phi_surf(sqp_problem.make_interp(init_phi))

  prob.set_coeffs(flow_norm=1e-9, flow_rigidity=1, obs=1, flow_agree=1)

  init_u = np.zeros(init_phi.shape + (2,))
  #init_u[:,:,0] = 1.
  out_phi_surf, out_u, opt_result = prob.optimize(init_phi, init_u, return_opt_result=True)
  plot_phi(221, sqp_problem.make_interp(init_phi))

  ax = fig.add_subplot(224)
  ax.plot(np.arange(len(opt_result.costs_over_iters)), opt_result.costs_over_iters)

  plot_phi(223, out_phi_surf)
  plot_u(222, out_u*10)

  print out_phi_surf.data
  print out_u

  plt.show()

if __name__ == '__main__':
  main()
