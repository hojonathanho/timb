import numpy as np
np.set_printoptions(linewidth=1000)
import timb

import matplotlib.pyplot as plt

def main_single_line():
  SIZE = 100
  phi = np.zeros((SIZE, SIZE))
  # phi[SIZE/4:SIZE*3/4, SIZE/2] = 0.
  phi[SIZE/4:SIZE*3/4, SIZE/2-1] = 1.
  phi[SIZE/4:SIZE*3/4, SIZE/2] = -1.
  ignore_mask = np.empty_like(phi, dtype=bool)
  ignore_mask.fill(True)
  ignore_mask[phi != 0] = False

  print np.count_nonzero(ignore_mask == False)

  # phi[SIZE/2,SIZE/2] = 0

  smoothed = timb.march_from_zero_crossing(phi, True, ignore_mask)
  # smoothed /= abs(smoothed).max()

  expected = np.array(
    [[ 6.5,  5.5,  4.5,  3.5,  2.5, -2.5, -3.5, -4.5, -5.5, -6.5],
     [ 5.5,  4.5,  3.5,  2.5,  1.5, -1.5, -2.5, -3.5, -4.5, -5.5],
     [ 4.5,  3.5,  2.5,  1.5,  0.5, -0.5, -1.5, -2.5, -3.5, -4.5],
     [ 4.5,  3.5,  2.5,  1.5,  0.5, -0.5, -1.5, -2.5, -3.5, -4.5],
     [ 4.5,  3.5,  2.5,  1.5,  0.5, -0.5, -1.5, -2.5, -3.5, -4.5],
     [ 4.5,  3.5,  2.5,  1.5,  0.5, -0.5, -1.5, -2.5, -3.5, -4.5],
     [ 4.5,  3.5,  2.5,  1.5,  0.5, -0.5, -1.5, -2.5, -3.5, -4.5],
     [ 5.5,  4.5,  3.5,  2.5,  1.5, -1.5, -2.5, -3.5, -4.5, -5.5],
     [ 6.5,  5.5,  4.5,  3.5,  2.5, -2.5, -3.5, -4.5, -5.5, -6.5],
     [ 7.5,  6.5,  5.5,  4.5,  3.5, -3.5, -4.5, -5.5, -6.5, -7.5]]
  )
  # assert np.allclose(smoothed, expected)

  print 'phi'
  print phi

  print 'mask'
  print ignore_mask

  print 'out'
  print smoothed

  plt.subplot(131)
  plt.title('Orig')
  plt.imshow(phi)

  # plt.subplot(312)
  # plt.imshow(smoothed, vmin=-1, vmax=1, cmap='bwr')

  plt.subplot(132)
  plt.title('FMM')
  plt.imshow(smoothed, cmap='bwr')

  plt.show()


from ctimbpy import *
def smooth_by_optimization(phi, obs, obs_weights, mode='laplacian'):
  assert phi.shape[0] == phi.shape[1]
  assert mode in ['laplacian', 'gradient', 'tps']

  print 'obs weights', obs_weights

  gp = GridParams(-1, 1, -1, 1, phi.shape[0], phi.shape[1])
  opt = Optimizer()
  # opt.params().check_linearizations = True
  phi_vars = make_var_field(opt, 'phi', gp)

  obs_cost = ObservationCost(phi_vars)
  obs_cost.set_observation(obs, obs_weights)
  opt.add_cost(obs_cost)

  if mode == 'laplacian':
    laplacian_cost = LaplacianCost(phi_vars)
    opt.add_cost(laplacian_cost, 1e-10)
  elif mode == 'gradient':
    grad_cost = GradientCost(phi_vars)
    opt.add_cost(grad_cost, 1e-10)
  elif mode == 'tps':
    tps_cost = TPSCost(phi_vars)
    opt.add_cost(tps_cost, 1e-10)
  else:
    raise NotImplementedError

  result = opt.optimize(phi.ravel())
  new_phi = result.x.reshape(phi.shape)

  return new_phi


def make_square_img(SIZE, negate_inside=True):
  a = np.empty((SIZE, SIZE), dtype=np.uint8); a.fill(255)
  a[int(SIZE/4.),int(SIZE/4.):-int(SIZE/4.)] = 0
  a[int(SIZE/4.):-int(SIZE/4.),int(SIZE/4.)] = 0
  a[int(SIZE/4.):-int(SIZE/4.),-int(SIZE/4.)] = 0
  a[-int(SIZE/4.),int(SIZE/4.):-int(SIZE/4.)+1] = 0
  img = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
  for i in range(3):
    img[:,:,i] = a
  return img


def main():
  SIZE = 100
  # phi = np.zeros((SIZE, SIZE))
  # phi.fill(np.inf)
  # obs_width = 6
  # obs = np.linspace(-1, 1, obs_width)
  # phi[SIZE/4:SIZE*3/4, SIZE/2-obs_width/2:SIZE/2+obs_width/2] = obs[None,:]
  # ignore_mask = np.empty_like(phi, dtype=bool)
  # ignore_mask.fill(True)
  # ignore_mask[phi != np.inf] = False
  # phi[phi == np.inf] = 0
  # obs_weights = (~ignore_mask).astype(float)


  orig_img = make_square_img(SIZE)
  angle = 0
  import observation
  import scipy.ndimage
  img = scipy.ndimage.interpolation.rotate(orig_img, angle, cval=255, order=1, reshape=False)
  state = (img[:,:,0] != 255) | (img[:,:,1] != 255) | (img[:,:,2] != 255)
  tsdf, sdf, depth = observation.state_to_tsdf(state, return_all=True)
  OBS_PEAK_WEIGHT = 1.
  obs_weights = observation.compute_obs_weight(sdf, depth, OBS_PEAK_WEIGHT)
  # truncated = abs(tsdf-sdf) < 1e-5
  truncated = abs(sdf) > observation.TRUNC_DIST
  obs_weights[truncated] = 0

  phi = tsdf
  ignore_mask = obs_weights == 0

  print 'phi'; print phi
  print 'mask'; print ignore_mask
  plt.subplot(331)
  plt.title('Orig')
  plt.imshow(phi, cmap='bwr', vmin=-observation.TRUNC_DIST, vmax=observation.TRUNC_DIST)
  # plt.plot(phi[50,:])
  plt.contour(phi, levels=[-10, 0, 10])

  # out_fmm = timb.march_from_zero_crossing(phi, True, ignore_mask)
  # print 'fmm'; print out_fmm
  # plt.subplot(332)
  # plt.title('FMM')
  # plt.imshow(out_fmm, cmap='bwr')

  plt.subplot(332)
  plt.title('Weights')
  plt.imshow(obs_weights, vmin=0, vmax=1, cmap='Greys_r')
  print 'weights'
  print obs_weights

  plt.subplot(333)
  plt.title('Weighted obs')
  plt.imshow(obs_weights*phi, cmap='bwr', vmin=-observation.TRUNC_DIST, vmax=observation.TRUNC_DIST)

  out_opt = smooth_by_optimization(phi, phi, obs_weights, mode='gradient')
  print 'gradient'; print out_opt
  v = SIZE
  plt.subplot(334)
  plt.title('Gradient cost')
  plt.imshow(out_opt, cmap='bwr', vmin=-20, vmax=20)
  plt.contour(out_opt, levels=[-10, 0, 10])


  out_opt = smooth_by_optimization(phi, phi, obs_weights, mode='laplacian')
  print 'laplacian'; print out_opt
  v = SIZE
  plt.subplot(335)
  plt.title('Laplacian cost')
  plt.imshow(out_opt, cmap='bwr', vmin=-20, vmax=20)
  plt.contour(out_opt, levels=[-10, 0, 10])
  # plt.plot(out_opt[50,:])

  out_opt = smooth_by_optimization(phi, phi, obs_weights, mode='tps')
  print 'tps'; print out_opt
  v = SIZE
  plt.subplot(336)
  plt.title('TPS cost')
  plt.imshow(out_opt, cmap='bwr', vmin=-20, vmax=20)
  plt.contour(out_opt, levels=[-10, 0, 10])
  # plt.plot(out_opt[50,:])

  plt.show()


if __name__ == '__main__':
  main()
