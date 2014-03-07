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
  phi = np.ones((size, size))
  w = np.zeros((size, size))

  phi[20:80,10:40] = np.linspace(1, -1, 40-10)[None,:]
  w[20:80,10:40] = 1.

  w[0,:] = 1.
  w[-1,:] = 1.
  w[:,0] = 1.
  w[:,-1] = 1.

  phi2 = timb.smooth(phi, w, 'tps')

  plt.subplot(141, aspect='equal')
  plot_field(phi, True, False)
  plt.axis('off')

  plt.subplot(142, aspect='equal')
  plot_field(phi, False, True)
  plt.axis('off')

  plt.subplot(143, aspect='equal')
  plot_field(phi2, True, False)
  plt.axis('off')

  plt.subplot(144, aspect='equal')
  plot_field(phi2, False, True)
  plt.axis('off')

  plt.savefig('out.pdf', bbox_inches='tight')

  plt.show()

if __name__ == '__main__':
  main()
