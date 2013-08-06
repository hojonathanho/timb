import flatland
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.morphology import distance_transform_edt

def show_vector_field(field_mn2, windowname):
  size = 10
  center_inds = np.dstack(np.meshgrid(np.arange(0, field_mn2.shape[0], size), np.arange(0, field_mn2.shape[1], size))).reshape((-1, 2))
  img = np.zeros((field_mn2.shape[0], field_mn2.shape[1], 3))
  img[center_inds] = (1., 1., 1.)
  cv2.imshow(windowname, img)

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
def test():
  n, m = 5, 4
  grid = np.random.rand(n, m)
  u = np.transpose(np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, m-1, m)))
  g2 = grid_interp(grid, u.reshape((-1, 2))).reshape((n, m))
  assert np.allclose(grid, g2)

  for j in range(1,m):
    g3 = grid_interp(grid, (u + [0,j]).reshape((-1, 2))).reshape((n, m))
    assert np.allclose(grid[:,j:], g3[:,:-j]) and np.allclose(g3[:,-1], grid[:,-1])

test()



def _eval_cost(size, obs, prev_sdf, sdf, u):
  ''' everything in image coordinates '''

  total = 0.

  # small flow
  flow_cost = (u**2).sum() / size/size
  print 'flow cost', flow_cost
  total += flow_cost

  # sdf and flow agree
  print u.shape
  shifted_x = np.transpose(np.meshgrid(np.linspace(0, size-1, size), np.linspace(0, size-1, size))) + u
  shifted_sdf = grid_interp(sdf, shifted_x.reshape((-1,2))).reshape(sdf.shape)
  agree_cost = ((shifted_sdf - prev_sdf)**2).sum() / size/size
  print 'agree', agree_cost
  total += agree_cost

  # sdf is zero at observation points
  sdf_at_obs = grid_interp(sdf, obs)
  obs_cost = (sdf_at_obs**2).sum() / size/size
  print 'obs cost', obs_cost
  total += obs_cost

  return total



class Tracker(object):
  def __init__(self, size, init_sdf):
    self.size = size
    self.sdf = init_sdf
    self.curr_u = np.zeros((size, size, 2))
    self.curr_obs = None

  def observe(self, obs_n2):
    self.curr_obs = obs_n2

  def eval_cost(self, prev_sdf, sdf, u):
    return _eval_cost(self.size, self.curr_obs, prev_sdf, sdf, u)

  def plot(self):
    #show_vector_field(self.curr_u, "u")

    cmap = np.zeros((256, 3),dtype='uint8')
    cmap[:,0] = range(256)
    cmap[:,2] = range(256)[::-1]
    cmap[0] = [0,0,0]
    flatland.show_2d_image(cmap[np.fmin((self.sdf*255).astype('int'), 255)], "sdf")

    cost = self.eval_cost(self.sdf, self.sdf, np.zeros((self.size, self.size, 2))+[0,1])
    print 'total cost', cost

SIZE = 200
# M_TO_PX = SIZE/2.
def main():
  #poly = flatland.Polygon([[.2, .2], [0,1], [1,1], [1,.5]])
  poly = flatland.Polygon([[0, 0], [1,0]])#, [1,1], [1,0]])
  polylist = [poly]
  cam_t = (0, -.5)
  r_angle = 0
  fov = 75 * np.pi/180.
  cam1d = flatland.Camera1d(cam_t, r_angle, fov, SIZE)
  cam2d = flatland.Camera2d((-1,-1), (1,1), SIZE)

  tracker = None

  while True:
    image1d, depth1d = cam1d.render(polylist)

    depth_min, depth_max = 0, 1
    depth1d_normalized = (np.clip(depth1d, depth_min, depth_max) - depth_min)/(depth_max - depth_min)
    depth1d_image = np.array([[.5, 0, 0]])*depth1d_normalized[:,None] + np.array([[1., 1., 1.]])*(1. - depth1d_normalized[:,None])
    depth1d_image[np.logical_not(np.isfinite(depth1d))] = (0, 0, 0)

    observed_XYs = cam1d.unproject(depth1d)
    filtered_obs_XYs = np.array([p for p in observed_XYs if np.isfinite(p).all()])

    # initialization
    if tracker is None:
      pass

    renderlist = polylist + [flatland.make_camera_poly(cam1d.t, cam1d.r_angle, fov)] + [flatland.Point(p, c) for (p, c) in zip(observed_XYs, depth1d_image) if np.isfinite(p).all()]
    image2d = cam2d.render(renderlist)

    flatland.show_1d_image([image1d, depth1d_image], "image1d+depth1d")
    flatland.show_2d_image(image2d)
    #flatland.show_2d_image(, "tracker_state")
    key = cv2.waitKey() & 255
    print "key", key

    # mac
    # if key == 63234:
    #     cam1d.r_angle -= .1
    # elif key == 63235:
    #     cam1d.r_angle += .1
    # elif key == 113:
    #     break

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
    elif key == ord('p'):

        # compute sdf of starting state as initialization
        image2d = cam2d.render(polylist)
        init_state_edge = np.ones((SIZE, SIZE), dtype=int)
        is_edge = image2d[:,:,0] > .5
        init_state_edge[is_edge] = 0
        init_sdf = distance_transform_edt(init_state_edge) / SIZE * 2.
        # negate inside the boundary
        orig_filling = [p.filled for p in polylist]
        for p in polylist: p.filled = True
        image2d_filled = cam2d.render(polylist)
        for orig, p in zip(orig_filling, polylist): p.filled = orig
        init_sdf[image2d_filled[:,:,0] > .5] *= -1

        tracker = Tracker(SIZE, init_sdf)
        P = flatland.Render2d(cam2d.bl, cam2d.tr, cam2d.width).P
        obs = np.fliplr((filtered_obs_XYs.dot(P[:2,:2].T) + P[:2,2][None,:]).astype(int))
        tracker.observe(obs)
        tracker.plot()

if __name__ == '__main__':
  main()
