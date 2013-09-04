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

# def zoom(m, factor):
#   import interpolation
#   s = interpolation.BilinearSurface(0, 1, 0, 1, m)
#   new_shape = (int(m.shape[0]*factor), int(m.shape[1]*factor))
#   new_is = np.linspace(0, m.shape[0]-1, new_shape[0])
#   new_js = np.linspace(0, m.shape[1]-1, new_shape[1])
#   ijs_up = np.c_[np.repeat(new_is, new_shape[1]), np.tile(new_js, new_shape[0])]
#   return s.eval_ijs(ijs_up).reshape(new_shape)

# def reintegrate(m, zero_thresh=1e-1, zoom_factor=100.):
#   zero_thresh += 1e-3
#   zm = zoom(m, zoom_factor)
#   print 'min', zm.min()
#   zero_mask = abs(zm) <= zero_thresh
#   zm[zero_mask] = 0
#   zm[np.logical_not(zero_mask)] = 1
#   smoothed = sn.morphology.distance_transform_edt(zm)
#   smoothed /= abs(smoothed).max()
#   return zoom(smoothed, 1./zoom_factor)

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

def rotate():
  empty_phi = np.ones((SIZE, SIZE))

  square_phi = np.ones_like(empty_phi)
  square_phi[int(SIZE/2.),int(SIZE/4.):-int(SIZE/4.)] = 0.
  square_phi[int(SIZE/2.):-int(SIZE/4.),int(SIZE/4.)] = 0.
  square_phi[int(SIZE/2.):-int(SIZE/4.),-int(SIZE/4.)] = 0.
  square_phi[-int(SIZE/4.),int(SIZE/4.):-int(SIZE/4.)+1] = 0.
  orig_phi = smooth(square_phi)

  init_phi = np.ones_like(empty_phi)
  init_obs_inds = calc_observable_inds(square_phi)
  init_phi[init_obs_inds[:,0],init_obs_inds[:,1]] = 0.
  init_phi = smooth(init_phi)

  # go through a rotation of the square
  prob = sqp_problem.TrackingProblem()
  prob.set_coeffs(flow_norm=1e-9, flow_rigidity=1, obs=1, flow_agree=1)
  for i_iter, angle in enumerate(np.arange(0, 359, 5)):
    if i_iter == 0:
      prob.set_coeffs(flow_agree=0)
    else:
      prob.set_coeffs(flow_agree=1)
    print 'Current angle:', angle
    # make an observation
    rotated = sn.interpolation.rotate(square_phi, angle, cval=1., order=0, reshape=False)
    print rotated.shape, square_phi.shape
    obs_inds = calc_observable_inds(rotated)
    prob.set_obs_points(obs_inds)
    prob.set_prev_phi(init_phi)

    init_u = np.zeros(init_phi.shape + (2,))
    out_phis, out_us, opt_result = prob.optimize(init_phi, init_u, return_full=True)


    for out_phi, out_u in [zip(out_phis, out_us)[-1]]:
      global fig
      fig = plt.figure()

      not_oob = (0 <= obs_inds[:,0]) & (obs_inds[:,0] < init_phi.shape[0]) & (0 <= obs_inds[:,1]) & (obs_inds[:,1] < init_phi.shape[1])
      init_phi[obs_inds[:,0][not_oob],obs_inds[:,1][not_oob]] = .5
      plot_phi(231, sqp_problem.make_interp(init_phi))

      ax = fig.add_subplot(234)
      tmp = rotated.copy()
      tmp[obs_inds[:,0],obs_inds[:,1]] = .5
      tmp = np.flipud(tmp.T)
      ax.imshow(tmp, cmap=cm.Greys_r).set_interpolation('nearest')
      # ax = fig.add_subplot(224)
      # ax.plot(np.arange(len(opt_result.costs_over_iters)), opt_result.costs_over_iters)

      plot_phi(232, sqp_problem.make_interp(out_phi))
      plot_u(233, out_u)

      # zero_thresh = sqp_problem.make_interp(out_phi).eval_ijs(obs_inds).mean()
      # print 'zero thresh', zero_thresh
      out_phi = reintegrate(out_phi)#, zero_thresh)
      plot_phi(235, sqp_problem.make_interp(out_phi))

      plt.show()


    # TODO: REINTEGRATE HERE
    init_phi = out_phi

    # raw_input('[Enter] for next rotation')


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

def rotate_full_obs():
  square_phi = make_square_phi()
  init_phi = square_phi.copy()

  # go through a rotation of the square
  prob = sqp_problem.TrackingProblem()
  prob.set_coeffs(flow_norm=1e-9, flow_rigidity=1e-3, flow_tps=1e-5, obs=1, flow_agree=1)
  for i_iter, angle in enumerate(np.arange(0, 359, 5)):
    print 'Current angle:', angle
    # make an observation
    rotated = sn.interpolation.rotate(square_phi, angle, cval=1., order=0, reshape=False)
    print rotated.shape, square_phi.shape
    obs_inds = calc_observable_inds(rotated)
    prob.set_obs_points(obs_inds)
    prob.set_prev_phi(init_phi)

    init_u = np.zeros(init_phi.shape + (2,))
    out_phis, out_us, opt_result = prob.optimize(init_phi, init_u, return_full=True)

    for out_phi, out_u in [zip(out_phis, out_us)[-1]]:
      global fig
      fig = plt.figure()

      # superimpose observation drawing
      not_oob = (0 <= obs_inds[:,0]) & (obs_inds[:,0] < init_phi.shape[0]) & (0 <= obs_inds[:,1]) & (obs_inds[:,1] < init_phi.shape[1])
      init_phi[obs_inds[:,0][not_oob],obs_inds[:,1][not_oob]] = 0
      plot_phi(231, sqp_problem.make_interp(init_phi))

      # ax = fig.add_subplot(234)
      # tmp = rotated.copy()
      # tmp[obs_inds[:,0],obs_inds[:,1]] = .5
      # tmp = np.flipud(tmp.T)
      # ax.imshow(tmp, cmap=cm.Greys_r).set_interpolation('nearest')
      ax = fig.add_subplot(234)
      ax.plot(np.arange(len(opt_result.costs_over_iters)), opt_result.costs_over_iters)

      plot_phi(232, sqp_problem.make_interp(out_phi))
      plot_u(233, out_u)

      out_phi = reintegrate(out_phi)
      plot_phi(235, sqp_problem.make_interp(out_phi))

      plt.show()

    init_phi = out_phi





  init_phi_observed = np.ones_like(empty_phi)
  init_phi_observed[int(SIZE/2.),int(SIZE/4.):-int(SIZE/4.)] = 0.

  new_phi_observed = init_phi_observed; new_phi_observed[int(SIZE/2.),int(SIZE/10.):-int(SIZE/10.)] = 0.
  obs_pts = np.transpose(np.nonzero(new_phi_observed == 0))

  prob = sqp_problem.TrackingProblem()
  prob.set_obs_points(obs_pts)
  prob.set_prev_phi(init_phi)

  prob.set_coeffs(flow_norm=1e-9, flow_rigidity=1e-8, flow_tps=1e-5, obs=1, flow_agree=1)

  init_u = np.zeros(init_phi.shape + (2,))
  #init_u[:,:,0] = 1.

  out_phis, out_us, opt_result = prob.optimize(init_phi, init_u, return_full=True)

  for out_phi, out_u in [zip(out_phis, out_us)[-1]]:
    # import cPickle
    # with open('/tmp/dump.pkl', 'w') as f:
    #   cPickle.dump((out_phi, out_u, opt_result), f)

    global fig
    fig = plt.figure(figsize=(15,15))

    init_phi[obs_pts[:,0],obs_pts[:,1]] = 0
    plot_phi(221, sqp_problem.make_interp(init_phi))

    ax = fig.add_subplot(224)
    ax.plot(np.arange(len(opt_result.costs_over_iters)), opt_result.costs_over_iters)

    out_phi = reintegrate(out_phi)
    plot_phi(223, sqp_problem.make_interp(out_phi))
    plot_u(222, out_u)

    plt.show()


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
  # new_phi_observed = sn.interpolation.shift(init_phi_observed, [0,-2], cval=1., order=0)
  # new_phi_observed = sn.interpolation.rotate(init_phi_observed, 5., cval=1., order=0)
  # plt.imshow(new_phi_observed, cmap=cm.Greys_r)
  # plt.show()
  obs_pts = np.transpose(np.nonzero(new_phi_observed == 0))

  prob = sqp_problem.TrackingProblem()
  prob.set_obs_points(obs_pts)
  prob.set_prev_phi(init_phi)

  prob.set_coeffs(flow_norm=1e-9, flow_rigidity=1e-8, flow_tps=1e-5, obs=1, flow_agree=1)

  init_u = np.zeros(init_phi.shape + (2,))
  #init_u[:,:,0] = 1.

  out_phis, out_us, opt_result = prob.optimize(init_phi, init_u, return_full=True)

  for out_phi, out_u in [zip(out_phis, out_us)[-1]]:
    # import cPickle
    # with open('/tmp/dump.pkl', 'w') as f:
    #   cPickle.dump((out_phi, out_u, opt_result), f)

    global fig
    fig = plt.figure(figsize=(15,15))

    init_phi[obs_pts[:,0],obs_pts[:,1]] = 0
    plot_phi(221, sqp_problem.make_interp(init_phi))

    ax = fig.add_subplot(224)
    ax.plot(np.arange(len(opt_result.costs_over_iters)), opt_result.costs_over_iters)

    out_phi = reintegrate(out_phi)
    plot_phi(223, sqp_problem.make_interp(out_phi))
    plot_u(222, out_u)

    plt.show()




  # for out_phi, out_u, opt_result in prob.optimize(init_phi, init_u, return_opt_result=True, yield_per_iter=True)():
  #   global fig
  #   fig = plt.figure()

  #   init_phi[obs_pts[:,0],obs_pts[:,1]] = .5
  #   plot_phi(221, sqp_problem.make_interp(init_phi))

  #   ax = fig.add_subplot(224)
  #   ax.plot(np.arange(len(opt_result.costs_over_iters)), opt_result.costs_over_iters)

  #   plot_phi(223, sqp_problem.make_interp(out_phi))
  #   plot_u(222, out_u*10)

  #   plt.show()

if __name__ == '__main__':
  rotate_full_obs()
