import numpy as np
import sympy
np.random.seed(0)

def gradient(u, eps_x=1, eps_y=1, wrt='xy', smooth_boundary=True):
  '''
  Gradients of scalar field
  Input: u (NxM)
  Output: gradient field (NxMx2) if wrt == 'xy', or just NxM if wrt == 'x' or 'y'
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
    g_x[1:-1,:] = (u[2:,:] - u[:-2,:]) / (2.*eps_x)
    if smooth_boundary:
      g_x[0,:] = (u[1,:] - u[0,:]) / (2.*eps_x)
      g_x[-1,:] = (u[-1,:] - u[-2,:]) / (2.*eps_x)
    else:
      g_x[0,:] = (u[1,:] - u[0,:]) / eps_x
      g_x[-1,:] = (u[-1,:] - u[-2,:]) / eps_x

  if wrt != 'x':
    g_y = g[:,:,1] if wrt == 'xy' else g
    g_y[:,1:-1] = (u[:,2:] - u[:,:-2]) / (2.*eps_y)
    if smooth_boundary:
      g_y[:,0] = (u[:,1] - u[:,0]) / (2.*eps_y)
      g_y[:,-1] = (u[:,-1] - u[:,-2]) / (2.*eps_y)
    else:
      g_y[:,0] = (u[:,1] - u[:,0]) / eps_y
      g_y[:,-1] = (u[:,-1] - u[:,-2]) / eps_y
 
  return g

class BicubicInterp(object):

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
    [ 4, -4, -4,  4,  2,  2, -2, -2,  2, -2,  2, -2,  1,  1,  1,  1]]
  )

  def __init__(self, xmin, xmax, ymin, ymax, data):
    self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
    self.data = data

    self.eps_x = (self.xmax-self.xmin)/(self.data.shape[0]-1.)
    self.eps_y = (self.ymax-self.ymin)/(self.data.shape[1]-1.)
    self.data_grad = gradient(self.data, self.eps_x, self.eps_y)
    self.data_grad_xy = gradient(self.data_grad[:,:,0], self.eps_x, self.eps_y, 'y')
    #assert np.allclose(self.deriv_xy, gradient(self.data_grad[:,:,1], wrt='x'))

    print 'computing interpolation weights'
    i0 = np.repeat(np.arange(0, self.data.shape[0] - 1), self.data.shape[1] - 1)
    j0 = np.tile(np.arange(0, self.data.shape[1] - 1), self.data.shape[0] - 1)
    i1, j1 = i0 + 1, j0 + 1
    X = np.c_[
      self.data[i0,j0], self.data[i1,j0], self.data[i0,j1], self.data[i1,j1],
      self.data_grad[i0,j0,0], self.data_grad[i1,j0,0], self.data_grad[i0,j1,0], self.data_grad[i1,j1,0],
      self.data_grad[i0,j0,1], self.data_grad[i1,j0,1], self.data_grad[i0,j1,1], self.data_grad[i1,j1,1],
      self.data_grad_xy[i0,j0], self.data_grad_xy[i1,j0], self.data_grad_xy[i0,j1], self.data_grad_xy[i1,j1]
    ]
    self.interp_coeffs = X.dot(self.bicubic_Ainv.T).reshape((self.data.shape[0] - 1, self.data.shape[1] - 1, 4, 4)).transpose((0, 1, 3, 2))

  def get_grid_coords(self):
    i = np.arange(0, self.data.shape[0], dtype=float)
    j = np.arange(0, self.data.shape[1], dtype=float)
    return self.to_xys(np.transpose(np.meshgrid(i, j)).reshape((-1, 2)))

  def to_xys(self, ijs):
    xys = np.empty_like(ijs)
    xys[:,0] = self.xmin + ijs[:,0]/(self.data.shape[0] - 1.)*(self.xmax - self.xmin)
    xys[:,1] = self.ymin + ijs[:,1]/(self.data.shape[1] - 1.)*(self.ymax - self.ymin)
    return xys

  def to_ijs(self, xys):
    ijs = np.empty_like(xys)
    ijs[:,0] = (xys[:,0] - self.xmin)*(self.data.shape[0] - 1.)/(self.xmax - self.xmin)
    ijs[:,1] = (xys[:,1] - self.ymin)*(self.data.shape[1] - 1.)/(self.ymax - self.ymin)
    return ijs

  def eval_single_ij(self, i, j):
    i0 = np.clip(int(i), 0, self.data.shape[0]-2)
    i1 = i0 + 1
    j0 = np.clip(int(j), 0, self.data.shape[1]-2)
    j1 = j0 + 1
    frac_i, frac_j = max(0, min(i-i0, 1)), max(0, min(j-j0, 1))

    basis_i = np.array([1, frac_i, frac_i**2, frac_i**3])
    basis_j = np.array([1, frac_j, frac_j**2, frac_j**3])
    val = (self.interp_coeffs[i0,j0,:,:] * basis_i[:,None] * basis_j[None,:]).sum()
    return val

  def eval_ijs(self, ijs):
    print ijs
    ijs = np.atleast_2d(ijs)
    num_pts = ijs.shape[0]

    int_ijs = ijs.astype(int)
    i0 = np.clip(int_ijs[:,0], 0, self.data.shape[0]-2)
    i1 = i0 + 1
    j0 = np.clip(int_ijs[:,1], 0, self.data.shape[1]-2)
    j1 = j0 + 1
    frac_i = np.clip(ijs[:,0] - i0, 0, 1)
    frac_j = np.clip(ijs[:,1] - j0, 0, 1)

    basis_i = np.c_[np.ones(num_pts), frac_i, frac_i**2, frac_i**3]
    basis_j = np.c_[np.ones(num_pts), frac_j, frac_j**2, frac_j**3]
    vals = (self.interp_coeffs[i0,j0,:,:] * basis_i[:,:,None] * basis_j[:,None,:]).sum(axis=1).sum(axis=1)
    return np.squeeze(vals)

  def eval_xys(self, xys):
    return self.eval_ijs(self.to_ijs(xys))



if __name__ == '__main__':
  rows, cols = 5, 5
  init_data = np.random.rand(rows, cols)
  assert np.allclose(np.dstack(np.gradient(init_data)), gradient(init_data, smooth_boundary=False))
  print 'ok'

  g = BicubicInterp(-1, 1, -2, 2, init_data)

  import matplotlib.cm as cm
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()

  res = 1000
  for z in np.linspace(-1, 1, 50):
    plt.plot(np.linspace(-3, 3, res), g.eval_xys(np.c_[z+np.zeros(res), np.linspace(-3, 3, res)]))
  plt.show()



  #plt.hold(true)
  ax = fig.add_subplot(111, projection='3d')
  assert np.allclose(g.eval_xys(g.get_grid_coords()), init_data.ravel())
  ax.scatter(g.get_grid_coords()[:,0], g.get_grid_coords()[:,1], g.eval_xys(g.get_grid_coords()))
  padding = 0
  res = 50
  X, Y = np.meshgrid(np.linspace(-1-padding, 1+padding, res), np.linspace(-2-padding, 2+padding, res))
  Z = g.eval_xys(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
  print X, Y, Z
  ax.plot_surface(X, Y, Z, antialiased=False)
  plt.show()




  import sympy
  mat = sympy.symarray('m', (rows, cols))
  sg = BicubicInterp(-1, 1, -2, 2, mat)
  import IPython; IPython.embed()
