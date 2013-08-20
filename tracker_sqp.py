import flatland
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.morphology import distance_transform_edt
import scipy.optimize
import solvers, grid
import sqp_problem
np.set_printoptions(linewidth=10000)

SIZE = 50
WORLD_MIN = (-1., -1.)
WORLD_MAX = (1., 1.)
PIXEL_AREA = 4./SIZE/SIZE
PIXEL_SIDE = 2./SIZE

sqp_problem.GRID_NX = SIZE
sqp_problem.GRID_NY = SIZE
sqp_problem.GRID_SHAPE = (SIZE, SIZE)
sqp_problem.GRID_MIN = WORLD_MIN
sqp_problem.GRID_MAX = WORLD_MAX
sqp_problem.PIXEL_AREA = PIXEL_AREA

def to_image_fmt(mat):
  assert mat.ndim == 2
  return np.flipud(np.fliplr(mat)).T

class Tracker(object):
  def __init__(self, init_phi):
    self.phi_surf = sqp_problem.make_bicubic(init_phi)
    self.curr_u = np.zeros(sqp_problem.GRID_SHAPE + (2,))
    self.curr_obs = None
    self.prob = None

    self.phi_cmap = np.zeros((256, 3), dtype='uint8')
    self.phi_cmap[:128,0] = 255
    self.phi_cmap[:128,1] = np.linspace(0, 255, 128).astype(int)
    self.phi_cmap[:128,2] = np.linspace(0, 255, 128).astype(int)
    self.phi_cmap[128:,0] = np.linspace(255, 0, 128).astype(int)
    self.phi_cmap[128:,1] = np.linspace(255, 0, 128).astype(int)
    self.phi_cmap[128:,2] = 255

  def observe(self, obs_n2):
    self.curr_obs = obs_n2

  def plot(self):
    # plot current sdf
    colors = np.clip((to_image_fmt(self.phi_surf.data)*100 + 128), 0, 255).astype(int)
    flatland.show_2d_image(self.phi_cmap[colors], 'phi')
    cv2.moveWindow('phi', 550, 0)

    # plot flow field
    grid.SpatialFunction(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], self.curr_u).show_as_vector_field('u')
    cv2.moveWindow("u", 0, 600)

  def optimize_sqp(self):
    if self.prob is None:
      self.prob = sqp_problem.TrackingProblem()
      self.prob.set_coeffs(flow_agree=0)

    self.prob.set_obs_points(self.curr_obs)
    self.prob.set_prev_phi_surf(self.phi_surf)

    self.phi_surf, self.curr_u = self.prob.optimize(np.zeros((SIZE,SIZE,2)))

    self.prob.set_coeffs(flow_agree=1)


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

  tracker = Tracker(np.ones((SIZE, SIZE)))

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

#   elif key == ord('c'):
#     tracker = Tracker(empty_sdf)
#     tracker.plot()
#     print 'zeroed out sdf and control'

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

      #obs_inds = np.c_[empty_sdf.to_grid_inds(obs[:,0], obs[:,1])].round()
      #print 'grid inds', obs_inds
      #obs = np.c_[empty_sdf.to_world_xys(obs_inds[:,0], obs_inds[:,1])]

      # print 'orig obs', obs
      #render2d = flatland.Render2d(cam2d.bl, cam2d.tr, cam2d.width)
      #xys = obs.dot(render2d.P[:2,:2].T) + render2d.P[:2,2]
      #ixys = xys.astype(int)
      #pts = []
      #for ix, iy in ixys:
      #  if 0 <= iy < render2d.image.shape[0] and 0 <= ix < render2d.image.shape[1]:
      #    pts.append([ix,iy])
      #print 'orig pts', pts
      #pts = np.array(pts)
      #obs = pts
      #Pinv = np.linalg.inv(render2d.P)
      #obs = np.array(pts).dot(Pinv[:2,:2].T) + Pinv[:2,2]
      #print 'rounded obs', obs
      #print 'rounded obs inds', empty_sdf.to_grid_inds(obs[:,0], obs[:,1])

      tracker.observe(obs)
      tracker.plot()
      print 'observed.'

    elif key == ord(' '):
      tracker.optimize_sqp()
      tracker.plot()

#     N = 1000
#     Z = tracker.phi_surf.eval_xys(np.c_[np.zeros(N), np.linspace(-1, 1, N)])
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.plot(np.linspace(-1, 1, N), Z)
#     plt.show()

if __name__ == '__main__':
  main()
