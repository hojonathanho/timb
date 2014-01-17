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


def smooth_by_laplacian(phi, obs, obs_weights):
  from ctimbpy import *
  assert phi.shape[0] == phi.shape[1]

  print 'obs weights', obs_weights

  gp = GridParams(-1, 1, -1, 1, phi.shape[0], phi.shape[1])
  opt = Optimizer()
  phi_vars = make_var_field(opt, 'phi', gp)

  obs_cost = ObservationCost(phi_vars)
  obs_cost.set_observation(obs, obs_weights)
  opt.add_cost(obs_cost)

  laplacian_cost = LaplacianCost(phi_vars)
  opt.add_cost(laplacian_cost, 1)

  result = opt.optimize(phi.ravel())
  new_phi = result.x.reshape(phi.shape)

  return new_phi



def main():
  SIZE = 20
  phi = np.zeros((SIZE, SIZE))
  # phi[SIZE/4:SIZE*3/4, SIZE/2] = 0.
  phi.fill(np.inf)
  obs_width = 10
  obs = np.linspace(-1, 1, obs_width)
  phi[SIZE/4:SIZE*3/4, SIZE/2-obs_width/2:SIZE/2+obs_width/2] = obs[None,:]
  ignore_mask = np.empty_like(phi, dtype=bool)
  ignore_mask.fill(True)
  ignore_mask[phi != np.inf] = False
  phi[phi == np.inf] = 0

  print 'phi'; print phi
  print 'mask'; print ignore_mask
  plt.subplot(131)
  plt.title('Orig')
  plt.imshow(phi, cmap='bwr')

  out_fmm = timb.march_from_zero_crossing(phi, True, ignore_mask)

  print 'fmm'; print out_fmm
  plt.subplot(132)
  plt.title('FMM')
  plt.imshow(out_fmm, cmap='bwr')

  out_opt = smooth_by_laplacian(phi, phi, (~ignore_mask).astype(float) + 1e-6)

  print 'opt'; print out_opt
  plt.subplot(133)
  plt.title('Optimization')
  plt.imshow(out_opt, cmap='bwr')

  plt.show()


if __name__ == '__main__':
  main()
