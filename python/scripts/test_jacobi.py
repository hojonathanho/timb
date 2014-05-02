import numpy as np
import timb
import matplotlib.pyplot as plt

def plot_field(f, img=True, contour=False):
  assert img or contour
  if img:
    plt.imshow(f.T, aspect=1, vmin=-1, vmax=1, cmap='bwr', origin='lower')
  else:
    plt.imshow(np.zeros_like(f), aspect=1, vmin=-1, vmax=1, cmap='bwr', origin='lower')
  if contour:
    x = np.linspace(0, f.shape[0]-1, f.shape[0])
    y = np.linspace(0, f.shape[1]-1, f.shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    plt.contour(X, Y, f, levels=[0], colors='k')

def main():
  size = 100
  world_min = (0., 0.)
  world_max = (size-1., size-1.)
  grid_params = timb.GridParams(world_min[0], world_max[0], world_min[1], world_max[1], size, size)

  phi_init = np.zeros((size,size))
  u_init = np.zeros((size,size))
  v_init = np.zeros((size,size))

  z = np.ones((size, size))
  w_z = np.zeros((size, size))
  z[20:80,10:40] = np.linspace(1, -1, 40-10)[None,:]
  w_z[20:80,10:40] = 1.

  mu_0 = np.zeros((size,size))
  mu_u = np.zeros((size,size))
  mu_v = np.zeros((size,size))
  wtilde = np.zeros((size,size))
  alpha = 0.
  beta = 0.

  gamma = 0.
  phi_0 = np.zeros((size,size))
  u_0 = np.zeros((size,size))
  v_0 = np.zeros((size,size))

  num_iters = 1000

  out_phi, out_u, out_v = timb.timb_solve_model_problem_jacobi(
    grid_params,
    phi_init, u_init, v_init,
    z, w_z,
    mu_0, mu_u, mu_v, wtilde,
    alpha, beta,
    gamma, phi_0, u_0, v_0,
    num_iters
  )

  print out_phi
  out_phi[np.isnan(out_phi)] = 0
  print out_phi
  print np.count_nonzero(out_phi)

  plt.subplot(141, aspect='equal')
  plot_field(out_phi, True, False)
  plt.axis('off')

  # plt.subplot(142, aspect='equal')
  # plot_field(phi, False, True)
  # plt.axis('off')

  # plt.subplot(143, aspect='equal')
  # plot_field(phi2, True, False)
  # plt.axis('off')

  # plt.subplot(144, aspect='equal')
  # plot_field(phi2, False, True)
  # plt.axis('off')

  # plt.savefig('out.pdf', bbox_inches='tight')

  plt.show()

if __name__ == '__main__':
  main()
