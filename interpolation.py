import numpy as np

def gradient(u, eps_x=1, eps_y=1, wrt='xy'):
  '''
  Gradients of scalar field
  Input: u (NxM)
  Output: gradient field (NxMx2) if wrt == 'xy', or just NxM if wrt == 'x' or 'y'

  Like np.gradient, except works with ndarrays of Python objects
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

def jacobian(u, eps_x=1, eps_y=1):
  '''
  Jacobians of a vector field
  '''
  out = np.empty((u.shape[0], u.shape[1], 2, 2), dtype=u.dtype)
  out[:,:,0,:] = gradient(u[:,:,0], eps_x, eps_y, wrt='xy')
  out[:,:,1,:] = gradient(u[:,:,1], eps_x, eps_y, wrt='xy')
  return out


class Surface(object):
  def __init__(self, xmin, xmax, ymin, ymax, data):
    assert data.ndim == 2

    self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
    self.eps_x = (self.xmax-self.xmin)/(data.shape[0]-1.)
    self.eps_y = (self.ymax-self.ymin)/(data.shape[1]-1.)
    self.nx, self.ny = data.shape[0], data.shape[1]

    x, y = np.arange(0, self.nx, dtype=float), np.arange(0, self.ny, dtype=float)
    self.grid_ijs = np.c_[np.repeat(x, self.ny), np.tile(y, self.nx)]
    #self.grid_ijs = np.transpose(np.meshgrid(np.arange(0, self.nx, dtype=float), np.arange(0, self.ny, dtype=float))).reshape((-1, 2))

    self.data = data

  def to_xys(self, ijs):
    ijs = np.atleast_2d(ijs)
    xys = np.empty_like(ijs)
    xys[:,0] = self.xmin + ijs[:,0]*self.eps_x
    xys[:,1] = self.ymin + ijs[:,1]*self.eps_y
    return xys

  def to_ijs(self, xys):
    xys = np.atleast_2d(xys)
    ijs = np.empty_like(xys)
    ijs[:,0] = (xys[:,0] - self.xmin)/self.eps_x
    ijs[:,1] = (xys[:,1] - self.ymin)/self.eps_y
    return ijs

  def get_grid_ijs(self):
    return self.grid_ijs

  def get_grid_xys(self):
    return self.to_xys(self.get_grid_ijs())

  def eval_ijs(self, ijs):
    raise NotImplementedError

  def eval_xys(self, xys):
    return self.eval_ijs(self.to_ijs(xys))

  def grad_ijs(self, ijs, delta=1e-5):
    orig_ijs = np.atleast_2d(ijs)
    ijs = orig_ijs.copy()

    grads = np.empty((ijs.shape[0], 2), dtype=self.data.dtype)
    for k in [0, 1]:
      ijs[:,k] = orig_ijs[:,k] + delta
      y1 = self.eval_ijs(ijs)
      ijs[:,k] = orig_ijs[:,k] - delta
      y0 = self.eval_ijs(ijs)
      ijs[:,k] = orig_ijs[:,k]
      grads[:,k] = (y1 - y0) / (2.*delta)
    return grads

  def grad_xys(self, xys):
    return self.grad_ijs(self.to_ijs(xys))



class BilinearSurface(Surface):
  def __init__(self, xmin, xmax, ymin, ymax, data):
    super(BilinearSurface, self).__init__(xmin, xmax, ymin, ymax, data)

  @staticmethod
  def grid_interp(grid, u):
    ''' bilinear interpolation '''
    assert u.shape[-1] == 2 and u.ndim == 2
    grid = np.atleast_3d(grid)

    ax, ay = np.floor(u[:,0]).astype(int), np.floor(u[:,1]).astype(int)
    bx, by = ax + 1, ay + 1
    ax, bx, ay, by = np.clip(ax, 0, grid.shape[0]-1), np.clip(bx, 0, grid.shape[0]-1), np.clip(ay, 0, grid.shape[1]-1), np.clip(by, 0, grid.shape[1]-1)

    # introduce axes into dx, dy to broadcast over all dimensions of output
    idx = tuple([slice(None)] + [None]*(grid.ndim - 2))
    dx, dy = np.maximum(u[:,0] - ax, 0)[idx], np.maximum(u[:,1] - ay, 0)[idx]

    out = (1.-dy)*((1.-dx)*grid[ax,ay] + dx*grid[bx,ay]) + dy*((1.-dx)*grid[ax,by] + dx*grid[bx,by])

    assert len(out) == len(u)
    return out

  @staticmethod
  def grid_interp_grad_nd(grid, u, eps=1e-5):
    grid = np.atleast_3d(grid)
    assert grid.ndim == 3 and grid.shape[2] == 1
    grid = np.squeeze(grid)

    grads = np.empty((len(u),) + grid.shape)
    for i in range(grid.shape[0]):
      for j in range(grid.shape[1]):
        orig = grid[i,j]
        grid[i,j] = orig + eps
        y2 = grid_interp(grid, u)
        grid[i,j] = orig - eps
        y1 = grid_interp(grid, u)
        grid[i,j] = orig
        grads[:,i,j] = np.squeeze(y2 - y1) / (2.*eps)
    return grads

  @staticmethod
  def grid_interp_grad(grid, u):
    '''gradients of interpolated points with respect to the grid'''
    grid = np.atleast_3d(grid)
    assert grid.ndim == 3 and grid.shape[2] == 1
    grid = np.squeeze(grid)

    ax, ay = np.floor(u[:,0]).astype(int), np.floor(u[:,1]).astype(int)
    bx, by = ax + 1, ay + 1

    in_range_axay = (ax >= 0) & (ax < grid.shape[0]) & (ay >= 0) & (ay < grid.shape[1])
    in_range_axby = (ax >= 0) & (ax < grid.shape[0]) & (by >= 0) & (by < grid.shape[1])
    in_range_bxay = (bx >= 0) & (bx < grid.shape[0]) & (ay >= 0) & (ay < grid.shape[1])
    in_range_bxby = (bx >= 0) & (bx < grid.shape[0]) & (by >= 0) & (by < grid.shape[1])

    dx, dy = np.maximum(u[:,0] - ax, 0), np.maximum(u[:,1] - ay, 0)

    grads = np.zeros((len(u),) + grid.shape)
    grads[in_range_axay,ax[in_range_axay],ay[in_range_axay]] = ((1.-dy)*(1.-dx))[in_range_axay]
    grads[in_range_bxay,bx[in_range_bxay],ay[in_range_bxay]] = ((1.-dy)*dx)[in_range_bxay]
    grads[in_range_axby,ax[in_range_axby],by[in_range_axby]] = (dy*(1.-dx))[in_range_axby]
    grads[in_range_bxby,bx[in_range_bxby],by[in_range_bxby]] = (dy*dx)[in_range_bxby]

    return grads

  def eval_ijs(self, ijs):
    ijs = np.atleast_2d(ijs)
    return np.squeeze(BilinearSurface.grid_interp(self.data, ijs))


class BicubicSurface(Surface):

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
    super(BicubicSurface, self).__init__(xmin, xmax, ymin, ymax, data)

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

    # lazily computed interpolation coefficients -- interp_coeffs_computed is the validity mask
    self.interp_coeffs = np.empty((self.padded_data.shape[0] - 1, self.padded_data.shape[1] - 1, 4, 4), dtype=data.dtype)
    self.interp_coeffs_computed = np.zeros((self.padded_data.shape[0] - 1, self.padded_data.shape[1] - 1), dtype=bool)

  def _compute_interp_coeffs_nocache(self, ijs):
    i0, j0 = ijs[:,0], ijs[:,1]
    i1, j1 = i0 + 1, j0 + 1
    X = np.c_[
      self.padded_data[i0,j0], self.padded_data[i1,j0], self.padded_data[i0,j1], self.padded_data[i1,j1],
      self.padded_data_grad[i0,j0,0], self.padded_data_grad[i1,j0,0], self.padded_data_grad[i0,j1,0], self.padded_data_grad[i1,j1,0],
      self.padded_data_grad[i0,j0,1], self.padded_data_grad[i1,j0,1], self.padded_data_grad[i0,j1,1], self.padded_data_grad[i1,j1,1],
      self.padded_data_grad_xy[i0,j0], self.padded_data_grad_xy[i1,j0], self.padded_data_grad_xy[i0,j1], self.padded_data_grad_xy[i1,j1]
    ]
    return X.dot(self.bicubic_Ainv.T).reshape((-1, 4, 4)).transpose((0, 2, 1))

  def _compute_interp_coeffs(self, ijs):
    i0, j0 = ijs[:,0], ijs[:,1]
    not_computed = np.logical_not(self.interp_coeffs_computed[i0,j0])
    if not_computed.any():
      not_computed_ijs = ijs[not_computed,:]
      new_coeffs = self._compute_interp_coeffs_nocache(not_computed_ijs)
      self.interp_coeffs[not_computed_ijs[:,0],not_computed_ijs[:,1],:,:] = new_coeffs
      self.interp_coeffs_computed[not_computed_ijs[:,0],not_computed_ijs[:,1]] = True
    return self.interp_coeffs[i0,j0]

  def eval_ijs(self, ijs):
    ijs = np.atleast_2d(ijs)

    int_ijs = np.floor(ijs).astype(int)
    i0 = np.clip(int_ijs[:,0], -1, self.nx-1)
    j0 = np.clip(int_ijs[:,1], -1, self.ny-1)
    frac_i = np.clip(ijs[:,0] - i0, 0, 1)
    frac_j = np.clip(ijs[:,1] - j0, 0, 1)

    num_pts = ijs.shape[0]
    basis_i = np.c_[np.ones(num_pts), frac_i, frac_i**2, frac_i**3]
    basis_j = np.c_[np.ones(num_pts), frac_j, frac_j**2, frac_j**3]

    coeffs = self._compute_interp_coeffs(np.c_[i0+1,j0+1])

    vals = (coeffs * basis_i[:,:,None] * basis_j[:,None,:]).sum(axis=1).sum(axis=1)
    return np.squeeze(vals)



def demo_bicubic():
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
    plt.plot(np.linspace(-4, 4, res), g.eval_xys(np.c_[z+np.zeros(res), np.linspace(-4, 4, res)]))
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

if __name__ == '__main__':
  demo_bicubic()
