import flatland
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.morphology import distance_transform_edt
np.set_printoptions(linewidth=10000)

def show_vector_field(origins, field_mn2, windowname):
  size = 10
  center_inds = np.dstack(np.meshgrid(np.arange(0, field_mn2.shape[0], size), np.arange(0, field_mn2.shape[1], size))).reshape((-1, 2))
  img = np.zeros((field_mn2.shape[0], field_mn2.shape[1], 3))
  img[center_inds[0],center_inds[1],:] = (1., 1., 1.)
  flatland.show_2d_image(img, windowname)

def grid_interp(grid, u):
  ''' bilinear interpolation '''
  assert u.shape[-1] == grid.ndim and u.ndim == 2
  ax, ay = np.floor(u[:,0]).astype(int), np.floor(u[:,1]).astype(int)
  bx, by = ax + 1, ay + 1
  ax, bx, ay, by = np.clip(ax, 0, grid.shape[0]-1), np.clip(bx, 0, grid.shape[0]-1), np.clip(ay, 0, grid.shape[1]-1), np.clip(by, 0, grid.shape[1]-1)
  dx = u[:,0] - ax
  dy = u[:,1] - ay
  out = (1.-dy)*((1.-dx)*grid[ax,ay] + dx*grid[bx,ay]) + dy*((1.-dx)*grid[ax,by] + dx*grid[bx,by])
  assert len(out) == len(u)
  return out

class BasicFunction(object):
  def __init__(self, xmin, xmax, ymin, ymax, init_f):
    # coordinate mapping: (xmin, ymin) -> (0, 0), (xmax, ymax) -> (init_f.shape[0]-1, init_f.shape[1]-1)
    self.xmin, self.xmax = xmin, xmax
    self.ymin, self.ymax = ymin, ymax
    self.f = np.atleast_3d(init_f)
    self.output_dim = self.f.shape[2]

  def to_inds(self, xs, ys):
    assert len(xs) == len(ys)# and (xs >= self.xmin).all() and (xs <= self.xmax).all() and (ys >= self.ymin).all() and (ys <= self.ymax).all()
    print 'converting', xs, ys
    ixs = (xs - self.xmin)*(self.f.shape[0] - 1.)/(self.xmax - self.xmin)
    iys = (ys - self.ymin)*(self.f.shape[1] - 1.)/(self.ymax - self.ymin)
    #import IPython; IPython.embed()
    return ixs, iys

  def eval_xys(self, xys):
    assert xys.shape[1] == 2
    return self.eval(xys[:,0], xys[:,1])

  def eval(self, xs, ys):
    ixs, iys = self.to_inds(xs, ys)
    float_inds = np.c_[ixs, iys]
    out = np.empty((len(xs), self.output_dim))
    for d in range(self.output_dim):
      out[:,d] = grid_interp(self.f[:,:,d], float_inds)
    return np.squeeze(out)

  def data(self):
    return np.squeeze(self.f)

  @classmethod
  def FromImage(cls, xmin, xmax, ymin, ymax, img):
    f = np.fliplr(np.flipud(img)).T
    return cls(xmin, xmax, ymin, ymax, f)

  def to_image_fmt(self):
    return np.flipud(np.fliplr(np.squeeze(self.f).T))

  def copy(self):
    return BasicFunction(self.xmin, self.xmax, self.ymin, self.ymax, self.f.copy())

  def flow(self, u):
    assert (u.xmin, u.xmax, u.ymin, u.ymax) == (self.xmin, self.xmax, self.ymin, self.ymax)
    assert u.f.shape[:2] == self.f.shape[:2] and u.f.shape[2] == 2

    grid = np.transpose(np.meshgrid(np.linspace(self.xmin, self.xmax, self.f.shape[0]), np.linspace(self.ymin, self.ymax, self.f.shape[1])))
    assert u.f.shape == grid.shape
    grid -= u.f

    new_f = self.eval_xys(grid.reshape((-1,2))).reshape(self.f.shape)
    return BasicFunction(self.xmin, self.xmax, self.ymin, self.ymax, new_f)

def run_tests():
  import unittest
  class Tests(unittest.TestCase):

    def test_grid_interp(self):
      n, m = 5, 4
      grid = np.random.rand(n, m)
      u = np.transpose(np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, m-1, m)))
      g2 = grid_interp(grid, u.reshape((-1, 2))).reshape((n, m))
      self.assertTrue(np.allclose(grid, g2))

      for j in range(1,m):
        g3 = grid_interp(grid, (u + [0,j]).reshape((-1, 2))).reshape((n, m))
        self.assertTrue(np.allclose(grid[:,j:], g3[:,:-j]))
        self.assertTrue(np.allclose(g3[:,-1], grid[:,-1]))

    def test_func_eval(self):
      data = np.random.rand(4, 5)
      f = BasicFunction(-1, 1,  6, 7, data)
      coords = np.transpose(np.meshgrid(np.linspace(-1, 1, 4), np.linspace(6, 7, 5)))
      vals = f.eval_xys(coords.reshape((-1,2))).reshape(data.shape)
      self.assertTrue(np.allclose(vals, data))

    def test_flow(self):
      data = np.random.rand(4, 5)
      f = BasicFunction(-1, 1,  6, 8, data)

      zero_flow = BasicFunction(-1, 1, 6, 8, np.zeros(data.shape + (2,)))
      f2 = f.flow(zero_flow)
      self.assertTrue(np.allclose(f2.data(), data))

      x_flow_data = np.zeros(data.shape + (2,))
      x_flow_data[:,:,0] = 2./3.
      x_flow = BasicFunction(-1, 1, 6, 8, x_flow_data)
      f3 = f.flow(x_flow)
      self.assertTrue(np.allclose(f3.data()[1:,:], data[:-1,:]))
      self.assertTrue(np.allclose(f3.data()[0,:], data[0,:]))


  suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
  unittest.TextTestRunner(verbosity=2).run(suite)

# def _eval_cost(T_pixel_world, obs, prev_sdf, sdf, u):
#   ''' everything in world coordinates '''

#   assert np.allclose(T_pixel_world[0,0], T_pixel_world[1,1]) # this code doesn't work otherwise
#   sdf_pixel_size = 1./abs(T_pixel_world[0,0])

#   total = 0.

#   # small flow
#   flow_cost = (u**2).sum() * sdf_pixel_size*sdf_pixel_size
#   print 'flow cost', flow_cost
#   total += flow_cost

#   # sdf and flow agree
#   pixel_grid = np.transpose(np.meshgrid(np.linspace(0, len(sdf)-1, len(sdf)), np.linspace(0, len(sdf)-1, len(sdf))))
#   shifted_pixels = pixel_grid + u.reshape((-1,2)).dot(T_pixel_world[:2,:2].T).reshape(pixel_grid.shape)
#   shifted_sdf = grid_interp(sdf, shifted_pixels.reshape((-1,2))).reshape(sdf.shape)
#   agree_cost = ((shifted_sdf - prev_sdf)**2).sum() * sdf_pixel_size*sdf_pixel_size
#   print 'agree', agree_cost
#   total += agree_cost

#   # sdf is zero at observation points
#   obs_px = obs.dot(T_pixel_world[:2,:2].T) + T_pixel_world[:2,2][None,:]
#   sdf_at_obs = grid_interp(sdf, obs_px)
#   obs_cost = (sdf_at_obs**2).sum() #* sdf_pixel_size*sdf_pixel_size
#   print 'obs cost', obs_cost
#   total += obs_cost

#   return total

def _eval_cost(pixel_area, obs_n2, prev_sdf, sdf, u):
  ''' everything in world coordinates '''

  total = 0.

  # small flow
  flow_cost = (u.data()**2).sum() * pixel_area
  print 'flow cost', flow_cost
  total += flow_cost

  # sdf and flow agree
  shifted_sdf = prev_sdf.flow(u)
  agree_cost = ((shifted_sdf.data() - sdf.data())**2).sum() * pixel_area
  print 'agree', agree_cost
  total += agree_cost

  # sdf is zero at observation points
  #obs_px = obs.dot(T_pixel_world[:2,:2].T) + T_pixel_world[:2,2][None,:]
  #sdf_at_obs = grid_interp(sdf, obs_px)
  sdf_at_obs = sdf.eval_xys(obs_n2)
  print 'sdf at obs', sdf_at_obs
  print 'sdf max', sdf.f.max()
  obs_cost = (sdf_at_obs**2).sum() * np.sqrt(pixel_area)
  print 'obs cost', obs_cost
  total += obs_cost

  return total


SIZE = 50
WORLD_MIN = (-1, -1)
WORLD_MAX = (1, 1)
PIXEL_AREA = 4./SIZE/SIZE
PIXEL_SIDE = 2./SIZE

class Tracker(object):
  def __init__(self, init_sdf):
    self.sdf = init_sdf
    self.prev_sdf = init_sdf.copy()
    self.curr_u = BasicFunction(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], np.zeros(self.sdf.f.shape[:2] + (2,)))#+[1,0])
    self.curr_obs = None

  def observe(self, obs_n2):
    self.curr_obs = obs_n2

  def eval_cost(self, sdf, u):
    return _eval_cost(PIXEL_AREA, self.curr_obs, self.prev_sdf, sdf, u)

  def plot(self):
    # plot current sdf
    curr_obs_inds = self.sdf.to_inds(self.curr_obs[:,0], self.curr_obs[:,1])
    print 'obs inds', curr_obs_inds
    print 'sdf\n', self.sdf.data()

    cmap = np.zeros((256, 3),dtype='uint8')
    cmap[:,0] = range(256)
    cmap[:,2] = range(256)[::-1]
    cmap[0] = [0,0,0]
    flatland.show_2d_image(cmap[np.fmin((self.sdf.to_image_fmt()*255).astype('int'), 255)], "sdf")

    # plot flow field
    #show_vector_field(self.curr_u, "u")

    cost = self.eval_cost(self.sdf, self.curr_u)
    print 'total cost', cost

  def optimize(self):
    self.prev_sdf = self.sdf

    dim_sdf, dim_u = self.sdf.size, self.curr_u.size
    def state_to_vec(sdf, u):
      return np.concatenate((sdf.data().flatten(), u.data().flatten()))
    def vec_to_state(vec):
      sdf = BasicFunction(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], vec[:dim_sdf].reshape(self.sdf.f.shape))
      u = BasicFunction(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], vec[dim_sdf:].reshape(self.curr_u.f.shape))
      return sdf, u

    import scipy.optimize
    self.i = 0
    def func(x):
      sdf, u = vec_to_state(x)
      cost = self.eval_cost(sdf, u)
      self.i += 1
      if self.i % 1000 == 0:
        self.plot()
        cv2.waitKey()
      return cost
    x0 = state_to_vec(self.prev_sdf, self.curr_u) # zero u?
    xopt = scipy.optimize.fmin_cg(func, x0)
    new_sdf, new_u = vec_to_state(xopt)

    # print 'Results'
    # print new_sdf
    # print new_u
    self.sdf, self.curr_u = new_sdf, new_u
    return new_sdf, new_u

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--test', action='store_true')
  args = parser.parse_args()

  if args.test:
    run_tests()

  poly = flatland.Polygon([[.2, .2], [0,1], [1,1], [1,.5]])
  #poly = flatland.Polygon([[0, 0], [1,0]])#, [1,1], [1,0]])
  polylist = [poly]
  cam_t = (0, -.5)
  r_angle = 0
  fov = 75 * np.pi/180.
  cam1d = flatland.Camera1d(cam_t, r_angle, fov, SIZE)
  cam2d = flatland.Camera2d(WORLD_MIN, WORLD_MAX, SIZE)

  tracker = None

  while True:
    image1d, depth1d = cam1d.render(polylist)

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
    flatland.show_2d_image(image2d)
    #flatland.show_2d_image(, "tracker_state")
    key = cv2.waitKey() & 255
    print "key", key

    # linux
    if key == 81:
        #cam1d.r_angle -= .1
        cam1d.t[0] += .1
    elif key == 82:
        cam1d.t[1] += .1
    elif key == 84:
        cam1d.t[1] -= .1
    elif key == 83:
        #cam1d.r_angle += .1
        cam1d.t[0] -= .1
    elif key == ord('q'):
        break

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
      init_sdf = BasicFunction.FromImage(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], init_sdf_img)

      tracker = Tracker(init_sdf)
      tracker.observe(filtered_obs_XYs)
      tracker.plot()
      cv2.waitKey()

    elif key == ord('o'):
      obs = filtered_obs_XYs
      tracker.observe(obs)
      print 'observed.'

    elif key == ord('p'):



        #T_pixel_world = flatland.Render2d(cam2d.bl, cam2d.tr, cam2d.width).P
        #T_pixel_world[:2,2] += .5
        #print 'T_pixel_world', T_pixel_world
        #tracker = Tracker(T_pixel_world, init_sdf)
        # obs = np.fliplr((filtered_obs_XYs.dot(P[:2,:2].T) + P[:2,2][None,:]).astype(int))

        for i in range(10000):
          #new_sdf, new_u = tracker.optimize()
          tracker.plot()
          cv2.waitKey()

if __name__ == '__main__':
  main()
