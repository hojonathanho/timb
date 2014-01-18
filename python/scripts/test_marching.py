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



def main():
  SIZE = 100
  phi = np.zeros((SIZE, SIZE))
  # phi[SIZE/4:SIZE*3/4, SIZE/2] = 0.
  phi.fill(np.inf)
  obs_width = 6
  obs = np.linspace(-1, 1, obs_width)
  phi[SIZE/4:SIZE*3/4, SIZE/2-obs_width/2:SIZE/2+obs_width/2] = obs[None,:]
  ignore_mask = np.empty_like(phi, dtype=bool)
  ignore_mask.fill(True)
  ignore_mask[phi != np.inf] = False
  phi[phi == np.inf] = 0

  print 'phi'; print phi
  print 'mask'; print ignore_mask
  plt.subplot(331)
  plt.title('Orig')
  plt.imshow(phi, cmap='bwr')

  out_fmm = timb.march_from_zero_crossing(phi, True, ignore_mask)
  print 'fmm'; print out_fmm
  plt.subplot(332)
  plt.title('FMM')
  plt.imshow(out_fmm, cmap='bwr')

  out_opt = smooth_by_optimization(phi, phi, (~ignore_mask).astype(float), mode='gradient')
  print 'gradient'; print out_opt
  v = SIZE
  plt.subplot(333)
  plt.title('Gradient cost')
  plt.imshow(out_opt, cmap='bwr')#, vmin=-20, vmax=20)


  out_opt = smooth_by_optimization(phi, phi, (~ignore_mask).astype(float), mode='laplacian')
  print 'laplacian'; print out_opt
  v = SIZE
  plt.subplot(334)
  plt.title('Laplacian cost')
  plt.imshow(out_opt, cmap='bwr')#, vmin=-20, vmax=20)
  # plt.plot(out_opt[50,:])

  out_opt = smooth_by_optimization(phi, phi, (~ignore_mask).astype(float), mode='tps')
  print 'tps'; print out_opt
  v = SIZE
  plt.subplot(335)
  plt.title('TPS cost')
  plt.imshow(out_opt, cmap='bwr')#, vmin=-20, vmax=20)
  # plt.plot(out_opt[50,:])

  plt.show()


if __name__ == '__main__':
  main()
