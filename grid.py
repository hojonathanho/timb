import numpy as np

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


class SpatialFunction(object):
  def __init__(self, xmin, xmax, ymin, ymax, f):
    # coordinate mapping: (xmin, ymin) -> (0, 0), (xmax, ymax) -> (f.shape[0]-1, f.shape[1]-1)
    self.xmin, self.xmax = xmin, xmax
    self.ymin, self.ymax = ymin, ymax
    self._f = np.atleast_3d(f)
    self.output_dim = self._f.shape[2:]

    self._df = None

  @classmethod
  def InitLike(cls, other, new_f):
    return cls(other.xmin, other.xmax, other.ymin, other.ymax, new_f)

  def _precompute_jacs(self):
    # if len(self.output_dim) != 1:
    #   return

    import itertools

    jac_f = np.empty(self._f.shape + (2,))
    for inds in itertools.product(*[range(d) for d in self.output_dim]):
    #for d in range(self.output_dim[0]):

      s = (slice(None), slice(None)) + inds

      jac_f[s + (slice(None),)] = np.dstack(np.gradient(self._f[s]))
      jac_f[s + (0,)] *= (self._f.shape[0] - 1.)/(self.xmax - self.xmin)
      jac_f[s + (1,)] *= (self._f.shape[1] - 1.)/(self.ymax - self.ymin)

      # jac_f[:,:,d,:] = np.dstack(np.gradient(self._f[:,:,d]))
      # jac_f[:,:,d,0] *= (self._f.shape[0] - 1.)/(self.xmax - self.xmin)
      # jac_f[:,:,d,1] *= (self._f.shape[1] - 1.)/(self.ymax - self.ymin)
    self._df = SpatialFunction(self.xmin, self.xmax, self.ymin, self.ymax, jac_f)

  def to_grid_inds(self, xs, ys):
    assert len(xs) == len(ys)
    ixs = (xs - self.xmin)*(self._f.shape[0] - 1.)/(self.xmax - self.xmin)
    iys = (ys - self.ymin)*(self._f.shape[1] - 1.)/(self.ymax - self.ymin)
    print ixs, iys
    return ixs, iys

  def to_world_xys(self, ixs, iys):
    xs = self.xmin + ixs.astype(float)/(self._f.shape[0] - 1.)*(self.xmax - self.xmin)
    ys = self.ymin + iys.astype(float)/(self._f.shape[1] - 1.)*(self.ymax - self.ymin)
    return xs, ys

  def _eval_inds(self, inds):
    return np.squeeze(grid_interp(self._f, inds))

  def eval_xys(self, xys):
    assert xys.shape[1] == 2
    return self.eval(xys[:,0], xys[:,1])

  def eval(self, xs, ys):
    ixs, iys = self.to_grid_inds(xs, ys)
    return self._eval_inds(np.c_[ixs, iys])

  def num_jac_direct(self, xs, ys, eps=1e-5):
    '''numerical jacobian, not using precomputed data'''
    #assert len(self.output_dim) == 1
    num_pts = len(xs); assert len(ys) == num_pts
    eps_x = np.empty(num_pts); eps_x.fill(2.*eps)
    eps_y = np.empty(num_pts); eps_y.fill(2.*eps)
    one_sided_mask_x = (xs-eps < self.xmin) | (xs+eps > self.xmax)
    one_sided_mask_y = (ys-eps < self.ymin) | (ys+eps > self.ymax)
    eps_x[one_sided_mask_x] = eps
    eps_y[one_sided_mask_y] = eps

    # introduce axes into eps_x, eps_y to broadcast over all dimensions of output
    idx = tuple([slice(None)] + [None]*(self._f.ndim - 2))
    eps_x, eps_y = eps_x[idx], eps_y[idx]

    jacs = np.empty([num_pts] + list(self.output_dim) + [2])
    jacs[:,:,0] = (self.eval(xs+eps, ys) - self.eval(xs-eps, ys)) / eps_x
    jacs[:,:,1] = (self.eval(xs, ys+eps) - self.eval(xs, ys-eps)) / eps_y
    return np.squeeze(jacs)

  def num_jac(self, xs, ys):
    '''numerical jacobian, precalculated'''
    if self._df is None: self._precompute_jacs()
    return self._df.eval(xs, ys)

  def jac_data(self):
    if self._df is None: self._precompute_jacs()
    return self._df.data()

  def jac_func(self):
    if self._df is None: self._precompute_jacs()
    return self._df

  def data(self): return np.squeeze(self._f)
  def size(self): return self._f.size
  def shape(self): return self.data().shape

  @classmethod
  def FromImage(cls, xmin, xmax, ymin, ymax, img):
    f = np.fliplr(np.flipud(img.T))
    return cls(xmin, xmax, ymin, ymax, f)

  def to_image_fmt(self):
    assert self.output_dim == (1,)
    return np.flipud(np.fliplr(np.squeeze(self._f))).T

  def copy(self):
    out = SpatialFunction(self.xmin, self.xmax, self.ymin, self.ymax, self._f.copy())
    # if self._df is not None:
    #   out._df = self._df.copy()
    return out

  def flow(self, u):
    assert (u.xmin, u.xmax, u.ymin, u.ymax) == (self.xmin, self.xmax, self.ymin, self.ymax)
    assert u._f.shape[:2] == self._f.shape[:2] and u._f.shape[2] == 2

    xy_grid = np.transpose(np.meshgrid(np.linspace(self.xmin, self.xmax, self._f.shape[0]), np.linspace(self.ymin, self.ymax, self._f.shape[1])))
    assert u._f.shape == xy_grid.shape
    xy_grid -= u._f

    new_f = self.eval_xys(xy_grid.reshape((-1,2))).reshape(self._f.shape)
    return SpatialFunction(self.xmin, self.xmax, self.ymin, self.ymax, new_f)


  def show_as_vector_field(self, windowname):
    import cv2, flatland
    assert self.output_dim == (2,)

    img_size = (500, 500)
    nx, ny = min(self._f.shape[0], 50), min(self._f.shape[1], 50)
    def to_img_inds(xys):
      xs, ys = xys[:,0], xys[:,1]
      ixs = (xs - self.xmin)*(img_size[0] - 1.)/(self.xmax - self.xmin)
      iys = (ys - self.ymin)*(img_size[1] - 1.)/(self.ymax - self.ymin)
      return np.c_[ixs, iys]
      #return np.c_[ixs.astype(int), iys.astype(int)]

    vec_starts = np.transpose(np.meshgrid(np.linspace(self.xmin, self.xmax, nx), np.linspace(self.ymin, self.ymax, ny)))
    vec_dirs = self.eval_xys(vec_starts.reshape((-1,2))).reshape((nx, ny, 2))
    vec_ends = vec_starts + vec_dirs

    img = np.zeros(img_size + (3,))
    for start, end in zip(to_img_inds(vec_starts.reshape((-1,2))), to_img_inds(vec_ends.reshape((-1,2)))):
      if np.linalg.norm(end - start) > 1e-5:
        cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), (1.,1.,1))
        d = (end - start) / np.linalg.norm(end - start)
        p = np.array([-d[1], d[0]])
        cv2.line(img, tuple(end.astype(int)), tuple((end - 5*d + 5*p).astype(int)), (1.,1.,1.))
        cv2.line(img, tuple(end.astype(int)), tuple((end - 5*d - 5*p).astype(int)), (1.,1.,1.))
      #cv2.circle(img, tuple(start), 1, (0,1.,0))
      img[int(start[1]), int(start[0])] = (0, 1., 0)
    flatland.show_2d_image(np.flipud(np.fliplr(img)), windowname)
