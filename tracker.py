import flatland
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.morphology import distance_transform_edt
import scipy.optimize
import solvers, grid
np.set_printoptions(linewidth=10000)


SIZE = 10
WORLD_MIN = (-1, -1)
WORLD_MAX = (1, 1)
PIXEL_AREA = 4./SIZE/SIZE
PIXEL_SIDE = 2./SIZE

class Tracker(object):
  def __init__(self, init_sdf):
    self.sdf = init_sdf
    self.prev_sdf = init_sdf.copy()
    self.curr_u = grid.SpatialFunction(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], np.zeros(self.sdf.shape() + (2,)))#+[1,0])
    self.curr_obs = None

  def reset_sdf(self, sdf):
    self.sdf = sdf
    self.prev_sdf = sdf.copy()

  def observe(self, obs_n2):
    self.curr_obs = obs_n2

  def eval_cost(self, sdf, u, return_full=False):
    return solvers._eval_cost(PIXEL_AREA, self.curr_obs, self.prev_sdf, sdf, u, return_full=return_full)

  def plot(self):
    # plot current sdf
    # curr_obs_inds = self.sdf.to_inds(self.curr_obs[:,0], self.curr_obs[:,1])
    # print 'obs inds', curr_obs_inds
    # print 'sdf\n', self.sdf.data()

    cmap = np.zeros((256, 3), dtype='uint8')
    cmap[:128,0] = 255
    cmap[:128,1] = np.linspace(0, 255, 128).astype(int)
    cmap[:128,2] = np.linspace(0, 255, 128).astype(int)
    cmap[128:,0] = np.linspace(255, 0, 128).astype(int)
    cmap[128:,1] = np.linspace(255, 0, 128).astype(int)
    cmap[128:,2] = 255

    # cmap[:,0] = range(256)
    # cmap[:,2] = range(256)[::-1]
    colors = np.clip((self.sdf.to_image_fmt()*100 + 128), 0, 255).astype(int)
    flatland.show_2d_image(cmap[colors], "sdf")
    cv2.moveWindow("sdf", 550, 0)

    # plot flow field
    # print 'curr flow\n', self.curr_u.data()
    self.curr_u.show_as_vector_field("u")
    cv2.moveWindow("u", 0, 600)

    total, costs = solvers._eval_cost(PIXEL_AREA, self.curr_obs, self.prev_sdf, self.sdf, self.curr_u, ignore_obs=self.curr_obs is None, return_full=True)
    print 'total cost', total
    print 'individual costs', costs

  def optimize(self):
    self.prev_sdf = self.sdf

    dim_sdf, dim_u = self.sdf.size(), self.curr_u.size()
    def state_to_vec(sdf, u):
      return np.concatenate((sdf.data().ravel(), u.data().ravel()))
    def vec_to_state(vec):
      assert len(vec) == dim_sdf + dim_u
      sdf = grid.SpatialFunction(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], vec[:dim_sdf].reshape(self.sdf.shape()))
      u = grid.SpatialFunction(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], vec[dim_sdf:].reshape(self.curr_u.shape()))
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
      return solvers._eval_cost_grad(PIXEL_AREA, self.curr_obs, self.prev_sdf, sdf, u)

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

    xopt = scipy.optimize.fmin_cg(func, x0, fprime=func_grad, disp=True, maxiter=10000)
    # import optimization
    # xopt = optimization.gradient_descent(func, func_grad, x0, maxiter=1000)
    new_sdf, new_u = vec_to_state(xopt)

    print 'Done optimizing.'
    # print new_sdf
    # print new_u
    print 'final costs:', self.eval_cost(new_sdf, new_u, return_full=True)
    print 'new sdf\n', new_sdf.data()
    self.sdf, self.curr_u = new_sdf, new_u
    #solvers.Weights.flow = 100.
    return new_sdf, new_u

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--test', action='store_true')
  args = parser.parse_args()

  if args.test:
    import tests
    tests.run_tests()

  cam_t = (0, -.5)
  r_angle = 0
  fov = 75 * np.pi/180.
  cam1d = flatland.Camera1d(cam_t, r_angle, fov, SIZE)
  cam2d = flatland.Camera2d(WORLD_MIN, WORLD_MAX, SIZE)

  empty_sdf = grid.SpatialFunction(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], np.ones((SIZE, SIZE)))
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
    # print 'depth1d', depth1d

    depth_min, depth_max = 0, 1
    depth1d_normalized = (np.clip(depth1d, depth_min, depth_max) - depth_min)/(depth_max - depth_min)
    depth1d_image = np.array([[.5, 0, 0]])*depth1d_normalized[:,None] + np.array([[1., 1., 1.]])*(1. - depth1d_normalized[:,None])
    depth1d_image[np.logical_not(np.isfinite(depth1d))] = (0, 0, 0)

    observed_XYs = cam1d.unproject(depth1d)
    filtered_obs_XYs = np.array([p for p in observed_XYs if np.isfinite(p).all()])

    obs_render_list = [flatland.Point(p, c) for (p, c) in zip(observed_XYs, depth1d_image) if np.isfinite(p).all()]
    # print 'obs world', filtered_obs_XYs
    camera_poly_list = [flatland.make_camera_poly(cam1d.t, cam1d.r_angle, fov)]
    image2d = cam2d.render(polylist + obs_render_list + camera_poly_list)

    flatland.show_1d_image([image1d, depth1d_image], "image1d+depth1d")
    flatland.show_2d_image(image2d, "image2d")
    cv2.moveWindow("image2d", 0, 0)
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

    if key == ord('a'):
        poly_center[0] += .01
    elif key == ord('w'):
        poly_center[1] += .01
    elif key == ord('s'):
        poly_center[1] -= .01
    elif key == ord('d'):
        poly_center[0] -= .01
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
      init_sdf = grid.SpatialFunction.FromImage(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], init_sdf_img)

      tracker = Tracker(init_sdf)
      tracker.observe(filtered_obs_XYs)
      tracker.plot()

    elif key == ord('o'):
      obs = filtered_obs_XYs
      # hack: ensure obs occurs only on grid

      obs_inds = np.c_[empty_sdf.to_grid_inds(obs[:,0], obs[:,1])].round()
      print 'grid inds', obs_inds
      obs = np.c_[empty_sdf.to_world_xys(obs_inds[:,0], obs_inds[:,1])]

      # print 'orig obs', obs
      # render2d = flatland.Render2d(cam2d.bl, cam2d.tr, cam2d.width)
      # xys = obs.dot(render2d.P[:2,:2].T) + render2d.P[:2,2]
      # ixys = xys.astype(int)
      # pts = []
      # for ix, iy in ixys:
      #   if 0 <= iy < render2d.image.shape[0] and 0 <= ix < render2d.image.shape[1]:
      #     pts.append([ix,iy])
      # print 'orig pts', pts
      # pts = np.array(pts)
      # obs = pts
      # Pinv = np.linalg.inv(render2d.P)
      # obs = np.array(pts).dot(Pinv[:2,:2].T) + Pinv[:2,2]
      # print 'rounded obs', obs
      # print 'rounded obs inds', empty_sdf.to_grid_inds(obs[:,0], obs[:,1])

      tracker.observe(obs)
      tracker.plot()
      print 'observed.'

    elif key == ord(' '):
      tracker.optimize()
      tracker.plot()

if __name__ == '__main__':
  main()
