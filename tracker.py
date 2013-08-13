import flatland
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.morphology import distance_transform_edt
import scipy.optimize
np.set_printoptions(linewidth=10000)

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
    return ixs, iys

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
    assert self.output_dim == (2,)

    img_size = (500, 500)
    nx, ny = self._f.shape[0], self._f.shape[1]
    def to_img_inds(xys):
      xs, ys = xys[:,0], xys[:,1]
      ixs = (xs - self.xmin)*(img_size[0] - 1.)/(self.xmax - self.xmin)
      iys = (ys - self.ymin)*(img_size[1] - 1.)/(self.ymax - self.ymin)
      return np.c_[ixs.astype(int), iys.astype(int)]

    vec_starts = np.transpose(np.meshgrid(np.linspace(self.xmin, self.xmax, nx), np.linspace(self.ymin, self.ymax, ny)))
    vec_dirs = self.eval_xys(vec_starts.reshape((-1,2))).reshape((nx, ny, 2))
    vec_ends = vec_starts + vec_dirs*10

    img = np.zeros(img_size + (3,))
    for start, end in zip(to_img_inds(vec_starts.reshape((-1,2))), to_img_inds(vec_ends.reshape((-1,2)))):
      cv2.line(img, tuple(start), tuple(end), (1.,1.,1))
      #cv2.circle(img, tuple(start), 1, (0,1.,0))
      img[int(start[1]), int(start[0])] = (0, 1., 0)
    flatland.show_2d_image(np.flipud(np.fliplr(img)), windowname)


def run_tests():
  import unittest
  class Tests(unittest.TestCase):

    def test_grid_interp(self):
      n, m = 5, 4
      data = np.random.rand(n, m)
      u = np.transpose(np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, m-1, m)))
      g2 = grid_interp(data, u.reshape((-1, 2))).reshape((n, m))
      self.assertTrue(np.allclose(data, g2))
      for j in range(1,m):
        g3 = grid_interp(data, (u + [0,j]).reshape((-1, 2))).reshape((n, m))
        self.assertTrue(np.allclose(data[:,j:], g3[:,:-j]))
        self.assertTrue(np.allclose(g3[:,-1], data[:,-1]))

      vector_data = np.random.rand(n, m, 3)
      g4 = grid_interp(vector_data, u.reshape((-1, 2))).reshape((n, m, 3))
      self.assertTrue(np.allclose(vector_data, g4))

      matrix_data = np.random.rand(n, m, 2, 3)
      g5 = grid_interp(matrix_data, u.reshape((-1, 2))).reshape((n, m, 2, 3))
      self.assertTrue(np.allclose(matrix_data, g5))

    def test_grid_interp_grad(self):
      n, m = 10, 15
      data = np.random.rand(n, m)
      u = np.transpose(np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, m-1, m))).reshape((-1,2))
      g1 = grid_interp_grad_nd(data, u)
      g2 = grid_interp_grad(data, u)
      self.assertTrue(np.allclose(g1, g2))

      u2 = np.random.rand(30, 2)
      u2[:,0] *= n-1; u[:,1] *= m-1
      g3 = grid_interp_grad_nd(data, u2)
      g4 = grid_interp_grad(data, u2)
      self.assertTrue(np.allclose(g3, g4))

    def test_func_eval(self):
      data = np.random.rand(4, 5)
      f = SpatialFunction(-1, 1,  6, 7, data)
      coords = np.transpose(np.meshgrid(np.linspace(-1, 1, 4), np.linspace(6, 7, 5)))
      vals = f.eval_xys(coords.reshape((-1,2))).reshape(data.shape)
      self.assertTrue(np.allclose(vals, data))

    def test_flow(self):
      data = np.random.rand(4, 5)
      f = SpatialFunction(-1, 1,  6, 8, data)

      zero_flow = SpatialFunction(-1, 1, 6, 8, np.zeros(data.shape + (2,)))
      f2 = f.flow(zero_flow)
      self.assertTrue(np.allclose(f2.data(), data))

      x_flow_data = np.zeros(data.shape + (2,))
      x_flow_data[:,:,0] = 2./3.
      x_flow = SpatialFunction(-1, 1, 6, 8, x_flow_data)
      f3 = f.flow(x_flow)
      self.assertTrue(np.allclose(f3.data()[1:,:], data[:-1,:]))
      self.assertTrue(np.allclose(f3.data()[0,:], data[0,:]))

    def test_num_jac(self):
      '''test jacobians on a function f(x) = [g_1(x), g_2(x), g_3(x)] where g_i are scaled rosenbrock functions'''
      def rosen(x):
        return (100.0*(x[:,1:]-x[:,:-1]**2.0)**2.0 + (1-x[:,:-1])**2.0).sum(axis=1)
      def rosen_der(x):
        xm = x[:,1:-1]
        xm_m1 = x[:,:-2]
        xm_p1 = x[:,2:]
        der = np.zeros_like(x)
        der[:,1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[:,0] = -400*x[:,0]*(x[:,1]-x[:,0]**2) - 2*(1-x[:,0])
        der[:,-1] = 200*(x[:,-1]-x[:,-2]**2)
        return der

      ndim = 3
      n, m = 50, 60
      xmin, xmax, ymin, ymax = -1, 1, 6, 8
      u = np.transpose(np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, m))).reshape((-1,2))
      xs, ys = u[:,0], u[:,1]
      data = rosen(u).reshape((n, m)); data = np.dstack([(k+1.)*data for k in range(ndim)])

      f = SpatialFunction(xmin, xmax, ymin, ymax, data)
      jacs = f.num_jac_direct(xs, ys)
      jacs2 = f.num_jac(xs, ys)
      self.assertTrue(np.allclose(jacs, jacs2))

      true_jac_base = rosen_der(u)
      true_jacs = np.empty_like(jacs)
      for k in range(ndim):
        true_jacs[:,k,:] = (k+1.)*true_jac_base
      self.assertTrue(np.absolute(true_jacs - jacs).max()/true_jacs.ptp() < .01)

      # second derivatives
      jac_fn = SpatialFunction(xmin, xmax, ymin, ymax, jacs.reshape((n, m, ndim, 2)))
      self.assertTrue(np.allclose(jac_fn.num_jac_direct(xs, ys), jac_fn.num_jac(xs, ys)))

  suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
  unittest.TextTestRunner(verbosity=2).run(suite)


class Weights:
  flow_norm = 1.
  flow = 1.
  rigidity = 10.
  obs = 1.

def _eval_cost(pixel_area, obs_n2, prev_sdf, sdf, u, ignore_obs=False, return_full=False):
  ''' everything in world coordinates '''

  total = 0.
  costs = {}

  # small flow
  flow_cost = Weights.flow_norm * pixel_area * (u.data()**2).sum()
  costs['flow'] = flow_cost
  total += flow_cost

  # smooth flow (small gradients)
  # flow_smooth_cost = pixel_area * (u.jac_data()**2).sum()
  # costs['flow_smooth'] = flow_smooth_cost
  # total += flow_smooth_cost

  #XXXXXXXXXXXXXXXXXXXXXXXX
  #import IPython; IPython.embed()
  # # sdf and flow agree
  # shifted_sdf = prev_sdf.flow(u)
  # agree_cost = ((shifted_sdf.data() - sdf.data())**2).sum() * pixel_area
  # # print 'agree', agree_cost
  # costs['agree'] = agree_cost
  # total += agree_cost
  #XXXXXXXXXXXXXXXXXXXXXXXXXXX

  # linearized optical flow
  agree_cost = Weights.flow * pixel_area * (((prev_sdf.jac_data() * u.data()).sum(axis=2) + sdf.data() - prev_sdf.data())**2).sum()
  costs['agree'] = agree_cost
  total += agree_cost

  # rigidity of flow
  ### FINITE STRAIN
  # J = (u.jac_data() + np.eye(2)[None,None,:,:]).reshape((-1, 2, 2))
  # JT = np.transpose(J, axes=(0, 2, 1))
  # products = (JT[:,:,:,None] * J[:,None,:,:]).sum(axis=2)
  # rigid_cost = pixel_area * ((products - np.eye(2)[None,:,:])**2).sum()
  ### INFINITESIMAL STRAIN

  J = u.jac_data()
  JT = np.transpose(J, axes=(0, 1, 3, 2))
  M = J + JT
  rigid_cost = Weights.rigidity * pixel_area * (M**2).sum()
  costs['rigid'] = rigid_cost
  total += rigid_cost

  # sdf is zero at observation points
  if not ignore_obs:
    sdf_at_obs = sdf.eval_xys(obs_n2)
    obs_cost = Weights.obs * np.sqrt(pixel_area) * (sdf_at_obs**2).sum()
    costs['obs'] = obs_cost
    total += obs_cost

  if return_full:
    return total, costs
  return total


import time
def _eval_cost_grad(pixel_area, obs_n2, prev_sdf, sdf, u, ignore_obs=False):
  t_start = time.time()

  grad = np.zeros(sdf.size() + u.size())
  grad_sdf, grad_u = grad[:sdf.size()], grad[sdf.size():]

  flow_cost_grad_u = Weights.flow_norm * pixel_area * 2.*u.data().ravel()
  grad_u += flow_cost_grad_u

  agree_cost = (prev_sdf.jac_data() * u.data()).sum(axis=2) + sdf.data() - prev_sdf.data()
  agree_cost_grad_sdf = Weights.flow * pixel_area * 2. * agree_cost.ravel()
  agree_cost_grad_u = Weights.flow * pixel_area * 2. * (agree_cost[:,:,None] * prev_sdf.jac_data()).ravel()
  grad_sdf += agree_cost_grad_sdf
  grad_u += agree_cost_grad_u

  J = u.jac_data()
  JT = np.transpose(J, axes=(0, 1, 3, 2))
  M = J + JT
  rigid_cost_grad_u = np.zeros(u.shape())
  # alternative loop implementation
  # for i in range(u.shape()[0]):
  #   for j in range(u.shape()[1]):
  #     A = 0. if i == 0 else M[i-1,j,0,0]
  #     B = 0. if i == u.shape()[0]-1 else M[i+1,j,0,0]
  #     C = 0. if j == 0 else M[i,j-1,0,1]
  #     D = 0. if j == u.shape()[1]-1 else M[i,j+1,0,1]
  #     z = 2. / (u.shape()[0] - 1.) * (u.xmax - u.xmin)
  #     a = (2. if i == 1 else 1.)*A
  #     b = (2. if i == u.shape()[0]-2 else 1.)*B
  #     c = (2. if j == 1 else 1.)*C
  #     d = (2. if j == u.shape()[1]-2 else 1.)*D
  #     e = 0
  #     if i == 0:
  #       e += -2.*M[i,j,0,0]
  #     elif i == u.shape()[0]-1:
  #       e += 2.*M[i,j,0,0]
  #     if j == 0:
  #       e += -2.*M[i,j,0,1]
  #     elif j == u.shape()[0]-1:
  #       e += 2.*M[i,j,0,1]
  #     rigid_cost_grad_u[i,j,0] = 4./z * (a - b + c - d + e)
  #     A = 0. if i == 0 else M[i-1,j,0,1]
  #     B = 0. if i == u.shape()[0]-1 else M[i+1,j,0,1]
  #     C = 0. if j == 0 else M[i,j-1,1,1]
  #     D = 0. if j == u.shape()[1]-1 else M[i,j+1,1,1]
  #     z = 2. / (u.shape()[1] - 1.) * (u.ymax - u.ymin)
  #     a = (2. if i == 1 else 1.)*A
  #     b = (2. if i == u.shape()[0]-2 else 1.)*B
  #     c = (2. if j == 1 else 1.)*C
  #     d = (2. if j == u.shape()[1]-2 else 1.)*D
  #     e = 0
  #     if i == 0:
  #       e += -2.*M[i,j,0,1]
  #     elif i == u.shape()[0]-1:
  #       e += 2.*M[i,j,0,1]
  #     if j == 0:
  #       e += -2.*M[i,j,1,1]
  #     elif j == u.shape()[0]-1:
  #       e += 2.*M[i,j,1,1]
  #     rigid_cost_grad_u[i,j,1] = 4./z * (a - b + c - d + e)
  A = np.zeros((u.shape()[0], u.shape()[1]))
  B = np.zeros((u.shape()[0], u.shape()[1]))
  C = np.zeros((u.shape()[0], u.shape()[1]))
  D = np.zeros((u.shape()[0], u.shape()[1]))
  E = np.zeros((u.shape()[0], u.shape()[1]))
  A[1:,:] = M[:-1,:,0,0]; A[1,:] *= 2.
  B[:-1,:] = M[1:,:,0,0]; B[-2,:] *= 2.
  C[:,1:] = M[:,:-1,0,1]; C[:,1] *= 2.
  D[:,:-1] = M[:,1:,0,1]; D[:,-2] *= 2.
  E[0,:] = -2.*M[0,:,0,0]
  E[-1,:] = 2.*M[-1,:,0,0]
  E[:,0] += -2.*M[:,0,0,1]
  E[:,-1] += 2.*M[:,-1,0,1]
  rigid_cost_grad_u[:,:,0] = 2.*(u.shape()[0] - 1.)/(u.xmax - u.xmin) * (A - B + C - D + E)
  A.fill(0.); B.fill(0.); C.fill(0.); D.fill(0.); E.fill(0.)
  A[1:,:] = M[:-1,:,0,1]; A[1,:] *= 2.
  B[:-1,:] = M[1:,:,0,1]; B[-2,:] *= 2.
  C[:,1:] = M[:,:-1,1,1]; C[:,1] *= 2.
  D[:,:-1] = M[:,1:,1,1]; D[:,-2] *= 2.
  E[0,:] = -2.*M[0,:,0,1]
  E[-1,:] = 2.*M[-1,:,0,1]
  E[:,0] += -2.*M[:,0,1,1]
  E[:,-1] += 2.*M[:,-1,1,1]
  rigid_cost_grad_u[:,:,1] = 2.*(u.shape()[1] - 1.)/(u.ymax - u.ymin) * (A - B + C - D + E)
  rigid_cost_grad_u *= Weights.rigidity * pixel_area
  grad_u += rigid_cost_grad_u.ravel()

  if not ignore_obs:
    sdf_at_obs = sdf.eval_xys(obs_n2)
    obs_cost_grad_sdf = Weights.obs * np.sqrt(pixel_area) * 2. * (sdf_at_obs[:,None,None] * grid_interp_grad(sdf.data(), np.c_[sdf.to_grid_inds(obs_n2[:,0], obs_n2[:,1])])).sum(axis=0).ravel()
    grad_sdf += obs_cost_grad_sdf


  print 'grad eval time: %f, grad norm: %f' % (time.time()-t_start, np.linalg.norm(grad))

  return grad


SIZE = 100
WORLD_MIN = (-1, -1)
WORLD_MAX = (1, 1)
PIXEL_AREA = 4./SIZE/SIZE
PIXEL_SIDE = 2./SIZE

class Tracker(object):
  def __init__(self, init_sdf):
    self.sdf = init_sdf
    self.prev_sdf = init_sdf.copy()
    self.curr_u = SpatialFunction(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], np.zeros(self.sdf.shape() + (2,)))#+[1,0])
    self.curr_obs = None

  def reset_sdf(self, sdf):
    self.sdf = sdf
    self.prev_sdf = sdf.copy()

  def observe(self, obs_n2):
    self.curr_obs = obs_n2

  def eval_cost(self, sdf, u, return_full=False):
    return _eval_cost(PIXEL_AREA, self.curr_obs, self.prev_sdf, sdf, u, return_full=return_full)

  def plot(self):
    # plot current sdf
    # curr_obs_inds = self.sdf.to_inds(self.curr_obs[:,0], self.curr_obs[:,1])
    # print 'obs inds', curr_obs_inds
    # print 'sdf\n', self.sdf.data()

    cmap = np.zeros((256, 3),dtype='uint8')
    cmap[:,0] = range(256)
    cmap[:,2] = range(256)[::-1]
    cmap[0] = [0,0,0]
    flatland.show_2d_image(cmap[np.fmin((np.clip(self.sdf.to_image_fmt(), 0, np.inf)*255).astype('int'), 255)], "sdf")
    cv2.moveWindow("sdf", 550, 0)

    # plot flow field
    # print 'curr flow\n', self.curr_u.data()
    self.curr_u.show_as_vector_field("u")
    cv2.moveWindow("u", 0, 600)

    total, costs = _eval_cost(PIXEL_AREA, self.curr_obs, self.prev_sdf, self.sdf, self.curr_u, ignore_obs=self.curr_obs is None, return_full=True)
    print 'total cost', total
    print 'individual costs', costs

  def optimize(self):
    self.prev_sdf = self.sdf

    dim_sdf, dim_u = self.sdf.size(), self.curr_u.size()
    def state_to_vec(sdf, u):
      return np.concatenate((sdf.data().ravel(), u.data().ravel()))
    def vec_to_state(vec):
      assert len(vec) == dim_sdf + dim_u
      sdf = SpatialFunction(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], vec[:dim_sdf].reshape(self.sdf.shape()))
      u = SpatialFunction(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], vec[dim_sdf:].reshape(self.curr_u.shape()))
      return sdf, u

    tmp_sdf, tmp_u = vec_to_state(state_to_vec(self.sdf, self.curr_u))
    assert np.allclose(tmp_sdf.data(), self.sdf.data()) and np.allclose(tmp_u.data(), self.curr_u.data())

    print 'number of variables:', dim_sdf + dim_u

    def func(x):
      sdf, u = vec_to_state(x)
      cost = self.eval_cost(sdf, u)
      return cost

    def func_grad(x):
      sdf, u = vec_to_state(x)
      return _eval_cost_grad(PIXEL_AREA, self.curr_obs, self.prev_sdf, sdf, u)

    def func_grad_nd(x, eps=1e-5):
      grad = np.empty_like(x)
      dx = np.eye(len(x)) * eps
      for i in range(len(x)):
        grad[i] = (func(x+dx[i]) - func(x-dx[i])) / (2.*eps)
      # assert np.allclose(func_grad(x), grad)
      return grad

    # rand_x = (np.random.rand(dim_sdf + dim_u) - .5) * 100
    # print 'numerical grad'
    # g1 = func_grad_nd(rand_x)
    # print 'analytical grad'
    # g2 = func_grad(rand_x)
    # import IPython; IPython.embed()

    #zero_u = self.curr_u.copy(); zero_u.data().fill(0)
    #x0 = state_to_vec(self.prev_sdf, zero_u) # zero u?
    x0 = state_to_vec(self.prev_sdf, self.curr_u)

    print 'initial costs:', self.eval_cost(self.prev_sdf, self.curr_u, return_full=True)
    print 'old sdf\n', self.prev_sdf.data()

    xopt = scipy.optimize.fmin_cg(func, x0, fprime=func_grad, disp=True, maxiter=2000)
    new_sdf, new_u = vec_to_state(xopt)

    print 'Done optimizing.'
    # print new_sdf
    # print new_u
    print 'final costs:', self.eval_cost(new_sdf, new_u, return_full=True)
    print 'new sdf\n', new_sdf.data()
    self.sdf, self.curr_u = new_sdf, new_u
    return new_sdf, new_u

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--test', action='store_true')
  args = parser.parse_args()

  if args.test:
    run_tests()

  cam_t = (0, -.5)
  r_angle = 0
  fov = 75 * np.pi/180.
  cam1d = flatland.Camera1d(cam_t, r_angle, fov, SIZE)
  cam2d = flatland.Camera2d(WORLD_MIN, WORLD_MAX, SIZE)

  empty_sdf = SpatialFunction(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], np.ones((SIZE, SIZE)))
  tracker = Tracker(empty_sdf)

  poly_center = np.array([0., 0.])
  poly_scale = .5
  poly_rot = 0.
  while True:
    poly_pts = np.array([[-0.5, -0.5], [ 0.5, -0.5], [ 0.5,  0.5], [-0.5,  0.5]]).dot(flatland.rotation2d(poly_rot).T)*poly_scale + poly_center[None,:]

    #poly = flatland.Polygon([[.2, .2], [0,1], [1,1], [1,.5]])
    poly = flatland.Polygon(poly_pts)
    polylist = [poly]

    image1d, depth1d = cam1d.render(polylist)
    print 'depth1d', depth1d

    depth_min, depth_max = 0, 1
    depth1d_normalized = (np.clip(depth1d, depth_min, depth_max) - depth_min)/(depth_max - depth_min)
    depth1d_image = np.array([[.5, 0, 0]])*depth1d_normalized[:,None] + np.array([[1., 1., 1.]])*(1. - depth1d_normalized[:,None])
    depth1d_image[np.logical_not(np.isfinite(depth1d))] = (0, 0, 0)

    observed_XYs = cam1d.unproject(depth1d)
    filtered_obs_XYs = np.array([p for p in observed_XYs if np.isfinite(p).all()])

    obs_render_list = [flatland.Point(p, c) for (p, c) in zip(observed_XYs, depth1d_image) if np.isfinite(p).all()]
    print 'obs world', filtered_obs_XYs
    camera_poly_list = [flatland.make_camera_poly(cam1d.t, cam1d.r_angle, fov)]
    image2d = cam2d.render(polylist + obs_render_list + camera_poly_list)

    flatland.show_1d_image([image1d, depth1d_image], "image1d+depth1d")
    flatland.show_2d_image(image2d, "image2d")
    cv2.moveWindow("image2d", 0, 0)
    #flatland.show_2d_image(, "tracker_state")
    key = cv2.waitKey() & 255
    print "key", key

    # linux
    if key == 81:
        cam1d.t[0] += .1
    elif key == 82:
        cam1d.t[1] += .1
    elif key == 84:
        cam1d.t[1] -= .1
    elif key == 83:
        cam1d.t[0] -= .1
    elif key == ord('['):
        cam1d.r_angle -= .1
    elif key == ord(']'):
        cam1d.r_angle += .1

    elif key == ord('q'):
        break

    elif key == ord('-'):
      poly_rot -= .1
    elif key == ord('='):
      poly_rot += .1

    elif key == ord('c'):
      tracker = Tracker(empty_sdf)
      tracker.plot()
      print 'zeroed out sdf and control'

    elif key == ord('i'):
      # initialization
      # compute sdf of starting state as initialization
      image2d = cam2d.render(polylist)
      init_state_edge = np.ones((SIZE, SIZE), dtype=int)
      is_edge = image2d[:,:,0] > .5
      init_state_edge[is_edge] = 0
      init_sdf_img = distance_transform_edt(init_state_edge) * PIXEL_SIDE
      # negate inside the boundary
      orig_filling = [p.filled for p in polylist]
      for p in polylist: p.filled = True
      image2d_filled = cam2d.render(polylist)
      for orig, p in zip(orig_filling, polylist): p.filled = orig
      init_sdf_img[image2d_filled[:,:,0] > .5] *= -1
      init_sdf = SpatialFunction.FromImage(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], init_sdf_img)

      tracker = Tracker(init_sdf)
      tracker.observe(filtered_obs_XYs)
      tracker.plot()

    elif key == ord('o'):
      obs = filtered_obs_XYs
      tracker.observe(obs)
      tracker.plot()
      print 'observed.'

    elif key == ord(' '):
      tracker.optimize()
      tracker.plot()




if __name__ == '__main__':
  main()
