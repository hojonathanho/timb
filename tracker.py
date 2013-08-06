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

class Tracker(object):
  def __init__(self, size, init_sdf):
    self.size = size
    self.sdf = init_sdf
    self.curr_u = np.zeros((size, size, 2))

  def observe(self, obs_n3):
    self.obs_n3 = obs_n3

  def eval_cost(self):
    total = 0.
    return total


  def plot(self):
    #show_vector_field(self.curr_u, "u")

    cmap = np.zeros((256, 3),dtype='uint8')
    cmap[:,0] = range(256)
    cmap[:,2] = range(256)[::-1]
    cmap[0] = [0,0,0]
    cv2.imshow("sdf", cmap[np.fmin((self.sdf*255).astype('int'), 255)])

def main():
  poly = flatland.Polygon([[.2, .2], [0,1], [1,1], [1,.5]])
  polylist = [poly]
  cam_t = (0, -.5)
  r_angle = 0
  fov = 75 * np.pi/180.
  SIZE = 500
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


    # initialization
    if tracker is None:
      image2d = cam2d.render(polylist)
      init_state_edge = np.ones((SIZE, SIZE), dtype=int)
      is_edge = image2d[:,:,0] > .5
      init_state_edge[is_edge] = 0
      # print is_edge
      # cv2.imshow('edge', np.dstack([init_state_edge, init_state_edge, init_state_edge]))
      init_sdf = distance_transform_edt(init_state_edge) / SIZE * 2.
      # negate inside the boundary
      orig_filling = [p.filled for p in polylist]
      for p in polylist: p.filled = True
      image2d_filled = cam2d.render(polylist)
      for orig, p in zip(orig_filling, polylist): p.filled = orig

      interior = np.zeros((SIZE, SIZE), dtype=int)
      interior[image2d_filled[:,:,0] > .5] = 1
      init_sdf[interior == 1] *= -1

      # diff = init_state_edge[1:,:] - init_state_edge[:-1,:]
      # crossing_inds = np.transpose(np.nonzero(diff))[::2,:]
      #flips = 
      #cs = np.cumsum(flips, axis=0)
      #cs = np.cumsum(init_state_edge[2:,:] - init_state_edge[:-2,:], axis=0)
      #print cs
      #cs = np.absolute(cs)
      #cv2.imshow('cs', np.dstack([cs, cs, cs]).astype(float))
      
      # print init_sdf

      # cmap = np.zeros((256, 3),dtype='uint8')
      # cmap[:,0] = range(256)
      # cmap[:,2] = range(256)[::-1]
      # cmap[0] = [0,0,0]
      # cv2.imshow("init_sdf", cmap[np.fmin((init_sdf*255*.064).astype('int'), 255)])
      #init_sdf = image1d()
      tracker = Tracker(SIZE, init_sdf)

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
        tracker.plot()

if __name__ == '__main__':
  main()
