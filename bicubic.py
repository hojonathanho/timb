import numpy as np
import sympy

def gradient(u, eps_x=1, eps_y=1, wrt='xy'):
  '''
  Gradients of scalar field
  Input: u (NxM)
  Output: gradient field (NxMx2) if wrt == 'xy', or just NxM if wrt == 'x' or 'y'

  Like np.gradient, except works with sympy
  '''
  if wrt == 'xy':
    out_shape = u.shape + (2,)
  elif wrt == 'x' or wrt == 'y':
    out_shape = u.shape
  else:
    raise RuntimeError

  g = np.empty(out_shape, dtype=u.dtype)

  if wrt != 'y':
    g_x = g[:,:,0] if wrt == 'xy' else g
    g_x[0,:] = (u[1,:] - u[0,:]) / eps_x
    g_x[1:-1,:] = (u[2:,:] - u[:-2,:]) / (2.*eps_x)
    g_x[-1,:] = (u[-1,:] - u[-2,:]) / eps_x

  if wrt != 'x':
    g_y = g[:,:,1] if wrt == 'xy' else g
    g_y[:,0] = (u[:,1] - u[:,0]) / eps_y
    g_y[:,1:-1] = (u[:,2:] - u[:,:-2]) / (2.*eps_y)
    g_y[:,-1] = (u[:,-1] - u[:,-2]) / eps_y
 
  return g

class BicubicSurface(object):

  bicubic_Ainv = np.array([
    [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [-3,  3,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 2, -2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -2, -1,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  1,  1,  0,  0],
    [-3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0, -3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0],
    [ 9, -9, -9,  9,  6,  3, -6, -3,  6, -6,  3, -3,  4,  2,  2,  1],
    [-6,  6,  6, -6, -3, -3,  3,  3, -4,  4, -2,  2, -2, -2, -1, -1],
    [ 2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0],
    [-6,  6,  6, -6, -4, -2,  4,  2, -3,  3, -3,  3, -2, -1, -2, -1],
    [ 4, -4, -4,  4,  2,  2, -2, -2,  2, -2,  2, -2,  1,  1,  1,  1]],
    dtype=float
  )

  def __init__(self, xmin, xmax, ymin, ymax, data):
    self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
    self.eps_x = (self.xmax-self.xmin)/(data.shape[0]-1.)
    self.eps_y = (self.ymax-self.ymin)/(data.shape[1]-1.)
    self.nx, self.ny = data.shape[0], data.shape[1]

    self.data = data

    self.padded_data = np.empty((data.shape[0]+2, data.shape[1]+2), dtype=data.dtype)
    self.padded_data[1:-1,1:-1] = data
    self.padded_data[0,1:-1] = data[0,:]
    self.padded_data[-1,1:-1] = data[-1,:]
    self.padded_data[1:-1,0] = data[:,0]
    self.padded_data[1:-1,-1] = data[:,-1]
    self.padded_data[0,0] = self.padded_data[0,1]
    self.padded_data[-1,0] = self.padded_data[-1,1]
    self.padded_data[0,-1] = self.padded_data[1,-1]
    self.padded_data[-1,-1] = self.padded_data[-2,-1]

    self.padded_data_grad = gradient(self.padded_data, self.eps_x, self.eps_y)
    self.padded_data_grad_xy = gradient(self.padded_data_grad[:,:,0], self.eps_x, self.eps_y, 'y')

    print 'computing interpolation weights'
    i0 = np.repeat(np.arange(0, self.padded_data.shape[0] - 1), self.padded_data.shape[1] - 1)
    j0 = np.tile(np.arange(0, self.padded_data.shape[1] - 1), self.padded_data.shape[0] - 1)
    i1, j1 = i0 + 1, j0 + 1
    X = np.c_[
      self.padded_data[i0,j0], self.padded_data[i1,j0], self.padded_data[i0,j1], self.padded_data[i1,j1],
      self.padded_data_grad[i0,j0,0], self.padded_data_grad[i1,j0,0], self.padded_data_grad[i0,j1,0], self.padded_data_grad[i1,j1,0],
      self.padded_data_grad[i0,j0,1], self.padded_data_grad[i1,j0,1], self.padded_data_grad[i0,j1,1], self.padded_data_grad[i1,j1,1],
      self.padded_data_grad_xy[i0,j0], self.padded_data_grad_xy[i1,j0], self.padded_data_grad_xy[i0,j1], self.padded_data_grad_xy[i1,j1]
    ]
    self.interp_coeffs = X.dot(self.bicubic_Ainv.T).reshape((self.padded_data.shape[0] - 1, self.padded_data.shape[1] - 1, 4, 4)).transpose((0, 1, 3, 2))

  def get_grid_ijs(self):
    i = np.arange(0, self.nx, dtype=float)
    j = np.arange(0, self.ny, dtype=float)
    return np.transpose(np.meshgrid(i, j)).reshape((-1, 2))

  def get_grid_xys(self):
    return self.to_xys(self.get_grid_ijs())

  def to_xys(self, ijs):
    xys = np.empty_like(ijs)
    xys[:,0] = self.xmin + ijs[:,0]/(self.nx - 1.)*(self.xmax - self.xmin)
    xys[:,1] = self.ymin + ijs[:,1]/(self.ny - 1.)*(self.ymax - self.ymin)
    return xys

  def to_ijs(self, xys):
    ijs = np.empty_like(xys)
    ijs[:,0] = (xys[:,0] - self.xmin)*(self.nx - 1.)/(self.xmax - self.xmin)
    ijs[:,1] = (xys[:,1] - self.ymin)*(self.ny - 1.)/(self.ymax - self.ymin)
    return ijs

  def eval_ijs(self, ijs):
    ijs = np.atleast_2d(ijs)
    num_pts = ijs.shape[0]

    int_ijs = np.floor(ijs).astype(int)
    i0 = np.clip(int_ijs[:,0], -1, self.nx-1)
    j0 = np.clip(int_ijs[:,1], -1, self.ny-1)
    i1 = i0 + 1
    j1 = j0 + 1
    frac_i = np.clip(ijs[:,0] - i0, 0, 1)
    frac_j = np.clip(ijs[:,1] - j0, 0, 1)

    basis_i = np.c_[np.ones(num_pts), frac_i, frac_i**2, frac_i**3]
    basis_j = np.c_[np.ones(num_pts), frac_j, frac_j**2, frac_j**3]

    coeffs = self.interp_coeffs[i0+1,j0+1,:,:]
    vals = (coeffs * basis_i[:,:,None] * basis_j[:,None,:]).sum(axis=1).sum(axis=1)
    return np.squeeze(vals)

  def eval_xys(self, xys):
    return self.eval_ijs(self.to_ijs(xys))



if __name__ == '__main__':
  np.random.seed(0)

  rows, cols = 5, 5
  init_data = np.random.rand(rows, cols)
  assert np.allclose(np.dstack(np.gradient(init_data)), gradient(init_data))
  print 'ok'

  g = BicubicSurface(-1, 1, -2, 2, init_data)

  import matplotlib.cm as cm
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()

  res = 100
  for z in np.linspace(-3, 3, 300):
    plt.plot(np.linspace(-3, 3, res), g.eval_xys(np.c_[z+np.zeros(res), np.linspace(-3, 3, res)]))
  plt.show()

# ax = fig.add_subplot(111, projection='3d')
# assert np.allclose(g.eval_xys(g.get_grid_xys()), init_data.ravel())
# ax.scatter(g.get_grid_xys()[:,0], g.get_grid_xys()[:,1], g.eval_xys(g.get_grid_xys()), s=100)
# padding = 2
# res = 100
# X, Y = np.meshgrid(np.linspace(-1-padding, 1+padding, res), np.linspace(-2-padding, 2+padding, res))
# Z = g.eval_xys(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
# ax.plot_surface(X, Y, Z, antialiased=False, rstride=1, cstride=1, cmap=cm.hot)
# plt.show()

  import sympy
  mat = sympy.symarray('m', (rows, cols))
  sg = BicubicSurface(-1, 1, -2, 2, mat)
  import IPython; IPython.embed()
