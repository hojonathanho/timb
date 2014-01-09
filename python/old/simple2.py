import numpy as np
import sqp_problem
import ctimbpy
import scipy.ndimage as sn
np.set_printoptions(linewidth=10000)

SIZE = 50
WORLD_MIN = (0., 0.)
WORLD_MAX = (SIZE-1., SIZE-1.)
sqp_problem.Config.set(SIZE, SIZE, WORLD_MIN, WORLD_MAX)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
fig = None

def plot_phi(N, phi, highlight_inds=None):
  x = np.linspace(WORLD_MIN[0], WORLD_MAX[0], phi.nx)
  y = np.linspace(WORLD_MIN[1], WORLD_MAX[1], phi.ny)
  # X, Y = np.meshgrid(x, y)
  pts = np.c_[np.repeat(x, phi.ny), np.tile(y, phi.nx)]
  Z = phi.eval_xys(pts).reshape((phi.nx, phi.ny)).T
  Z = np.flipud(Z)

  ax = fig.add_subplot(N, aspect='equal')
  X,Y = np.meshgrid(x, y)
  ax.contour(X, Y, Z, levels=[0.])
  # if highlight_inds is None:
  ax.imshow(Z, cmap='RdBu').set_interpolation('nearest')

  # else:
  #   ZZ = 


  # ax.scatter(pts[:,0], pts[:,1], c=phi.eval_xys(pts), s=100, cmap=cm.Greys_r)

  # ax = fig.add_subplot(N, projection='3d')
  # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

def plot_u(N, u):
  x = np.linspace(WORLD_MIN[0], WORLD_MAX[0], u.shape[0])
  y = np.linspace(WORLD_MIN[1], WORLD_MAX[1], u.shape[1])
  Y, X = np.meshgrid(x, y)
  ax = fig.add_subplot(N, aspect='equal')
  ax.quiver(X, Y, u[:,:,0], u[:,:,1], angles='xy', scale_units='xy', scale=1.)

def smooth(phi):
  d = sn.morphology.distance_transform_edt(phi)
  d /= abs(d).max()
  return d

def reintegrate(m):
  import skfmm
  d = skfmm.distance(m)
  d /= abs(d).max()
  return d

def calc_observable_inds(bin_phi):
  a, b = np.nonzero(bin_phi.T == 0)
  inds = np.transpose((b,a))
  inds2 = np.empty_like(inds)
  inds2[:,0] = inds[:,1]
  inds2[:,1] = inds[:,0]
  return inds[np.r_[True, inds2[1:,0] != inds2[:-1,0]]]




def make_square_phi(negate_inside=True):
  square_phi = np.ones((SIZE, SIZE))
  square_phi[int(SIZE/2.),int(SIZE/4.):-int(SIZE/4.)] = 0.
  square_phi[int(SIZE/2.):-int(SIZE/4.),int(SIZE/4.)] = 0.
  square_phi[int(SIZE/2.):-int(SIZE/4.),-int(SIZE/4.)] = 0.
  square_phi[-int(SIZE/4.),int(SIZE/4.):-int(SIZE/4.)+1] = 0.
  square_phi = smooth(square_phi)
  if negate_inside:
    square_phi[int(SIZE/2.):-int(SIZE/4.),int(SIZE/4.):-int(SIZE/4.)] *= -1
  return square_phi



def simple():

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
  # negate inside
  init_phi[int(SIZE/2.):-int(SIZE/4.),int(SIZE/4.):-int(SIZE/4.)] *= -1

  # import cPickle
  # with open('/tmp/init_phi.pkl', 'w') as f: cPickle.dump(init_phi, f)

  init_phi_observed = np.ones_like(empty_phi)
  init_phi_observed[int(SIZE/2.),int(SIZE/4.):-int(SIZE/4.)] = 0.
  # init_phi_observed = smooth(init_phi_observed)


  new_phi_observed = init_phi_observed; new_phi_observed[int(SIZE/2.),int(SIZE/10.):-int(SIZE/10.)] = 0.
  # new_phi_observed = sn.interpolation.zoom(init_phi_observed, 1.2, cval=1., order=0)
  # new_phi_observed = sn.interpolation.shift(init_phi_observed, [2,0], cval=1., order=0)
  # new_phi_observed = sn.interpolation.rotate(init_phi_observed, 5., cval=1., order=0)
  # plt.imshow(new_phi_observed, cmap=cm.Greys_r)
  # plt.show()
  obs_pts = np.transpose(np.nonzero(new_phi_observed == 0))

  # prob = sqp_problem.TrackingProblem()
  prob = ctimbpy.TrackingProblem(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], SIZE, SIZE)
  prob.set_observation_points(obs_pts)
  prob.set_prev_phi(init_phi)

  # prob.set_coeffs(flow_norm=1e-9, flow_rigidity=1e-3, flow_tps=0, obs=1, flow_agree=1)

  init_u = np.zeros(init_phi.shape + (2,))
  #init_u[:,:,0] = 1.

  # out_phis, out_us, opt_result = prob.optimize(init_phi, init_u, return_full=True)
  result = prob.optimize()
  out_phis = [result.phi]
  out_us = [result.u]
  opt_result = result.opt_result

  for out_phi, out_u in [zip(out_phis, out_us)[-1]]:
    # import cPickle
    # with open('/tmp/dump.pkl', 'w') as f:
    #   cPickle.dump((out_phi, out_u, opt_result), f)

    global fig
    fig = plt.figure(figsize=(15,15))

    init_phi[obs_pts[:,0],obs_pts[:,1]] = 0
    plot_phi(221, sqp_problem.make_interp(init_phi))

    ax = fig.add_subplot(224)
    ax.plot(np.arange(len(opt_result.cost_over_iters)), opt_result.cost_over_iters)

    #out_phi = reintegrate(out_phi)
    plot_phi(223, sqp_problem.make_interp(out_phi))
    plot_u(222, out_u)

    plt.show()


if __name__ == '__main__':
  simple()
