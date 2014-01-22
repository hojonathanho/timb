import numpy as np
import skfmm

# states are boolean 2d arrays. false = free space, 1 = object

OBS_PEAK_WEIGHT = 1.

def _state_to_pixel_depth(state):
  '''
  State is a NxM boolean array. True indicates the presence of a surface.
  Returns an array of length N that contains the distance from the y=0 axis to the surface.
  '''
  depth = np.empty(state.shape[0])
  depth.fill(np.inf)
  for i in range(state.shape[0]):
    # look for first occurrence of surface
    for j in range(state.shape[1]):
      if state[i,j]:
        depth[i] = j
        break
  return depth


def _state_to_visibility_mask(state, depth=None):
  # make mask where visible surface is zero
  if depth is None:
    depth = _state_to_pixel_depth(state)
  mask = np.ones_like(state, dtype=float)
  for i in range(state.shape[0]):
    d = depth[i]
    if d != np.inf:
      # if the ray hits something, make a pixel zero if any neighbor is zero
      for j in range(int(d)):
        try:
          if state[i,j] or state[i+1,j] or state[i-1,j] or state[i,j+1] or state[i,j-1]:
            mask[i,j] = 0
        except IndexError:
          pass
  return mask


def _depth_to_weights(depth, filter_radius):
  '''
  Computes weights for a depth image
  '''

  has_measurement = (depth != np.inf).astype(float)
  filt = np.ones(2*filter_radius + 1); filt /= filt.sum()
  out = np.convolve(has_measurement, filt, 'same')

  def f(x):
    '''maps [.5, 1] to [0, 1], differentiable at boundary'''
    y = .5*np.sin(2*np.pi*(x - .75)) + .5
    y[x < .5] = 0.
    y[x > 1.] = 1.
    return y

  return f(out)


def state_to_tsdf(state, trunc_dist, mode='accurate', return_all=False):
  '''
  '''

  assert mode in ['accurate', 'projective']

  # observe from the left
  depth = _state_to_pixel_depth(state)

  if mode == 'accurate':
    # make mask where visible surface is zero
    mask = _state_to_visibility_mask(state, depth)
    # fill to make a SDF
    sdf = skfmm.distance(mask)
    for i in range(state.shape[0]):
      if depth[i] == np.inf:
        sdf[i,:] = np.inf
    # make region behind surface negative
    for i in range(state.shape[0]):
      d = depth[i]
      if d != np.inf:
        sdf[i,d+1:] *= -1

  elif mode == 'projective':
    # projective point-to-point metric
    # fill in sdf by marching away from surface
    sdf = np.zeros_like(state, dtype=float)
    for i in range(state.shape[0]):
      d = depth[i]
      if d == np.inf:
        sdf[i,:] = np.inf
      else:
        for j in range(state.shape[1]):
          sdf[i,j] = d - j

  else:
    raise NotImplementedError

  tsdf = np.clip(sdf, -trunc_dist, trunc_dist)

  if return_all: return tsdf, sdf, depth
  return tsdf


def compute_obs_weight(obs_sdf, depth, epsilon, delta, filter_radius):
  # a = abs(obs_tsdf)
  # return np.where(a < trunc_val, (OBS_PEAK_WEIGHT/trunc_val)*(trunc_val-a), 0.)

  # linear weight (b) in Bylow et al 2013
  # epsilon, delta = 5, 10
  # epsilon, delta = 0, 5
  assert delta > epsilon
  w = np.where(obs_sdf >= -epsilon, OBS_PEAK_WEIGHT, obs_sdf)
  w = np.where((obs_sdf <= -epsilon) & (obs_sdf >= -delta), (OBS_PEAK_WEIGHT/(delta-epsilon))*(obs_sdf + delta), w)
  w = np.where(obs_sdf < -delta, 0, w)
  w = np.where(np.isfinite(obs_sdf), w, 0) # zero precision where we get inf/no depth measurement

  dw = _depth_to_weights(depth, filter_radius)
  w *= dw[:,None]

  return w


########## TESTS ##########

def test_simple():
  import matplotlib.pyplot as plt

  state = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
  ], dtype=bool)
  tsdf = state_to_tsdf(state)
  #plt.imshow(state)
  # plt.contourf(tsdf, cmap='bwr')
  print tsdf
  # plt.show()

  assert np.allclose(tsdf, np.array(
    [[ 10.,  10.,  10.,  10.,  10.],
     [  1.,   0.,  -1.,  -2.,  -3.],
     [  1.,   0.,  -1.,  -2.,  -3.],
     [  1.,   0.,  -1.,  -2.,  -3.],
     [ 10.,  10.,  10.,  10.,  10.]]
  ))


def test_load_img():
  import matplotlib.pyplot as plt
  from scipy import ndimage

  img = ndimage.imread('/Users/jonathan/code/timb/data/smallbear2.png')

  img = ndimage.interpolation.rotate(img, angle=180, cval=255, order=1, reshape=False)
  plt.imshow(img).set_interpolation('nearest')

  state = (img[:,:,0] != 255) | (img[:,:,1] != 255) | (img[:,:,2] != 255)
  plt.figure(1)
  plt.contourf(state)

  tsdf = state_to_tsdf(state)
  print np.count_nonzero(abs(tsdf) < 10), tsdf.size
  plt.figure(2)
  plt.axis('equal')
  plt.contour(tsdf, levels=[0])
  plt.imshow(tsdf, cmap='bwr').set_interpolation('nearest')

  plt.show()


def test_rotate():
  from scipy import ndimage
  import matplotlib
  import matplotlib.pyplot as plt

  def make_square_img(size):
    a = np.empty((size, size), dtype=np.uint8); a.fill(255)
    a[int(size/4.),int(size/4.):-int(size/4.)] = 0
    a[int(size/4.):-int(size/4.),int(size/4.)] = 0
    a[int(size/4.):-int(size/4.),-int(size/4.)] = 0
    a[-int(size/4.),int(size/4.):-int(size/4.)+1] = 0
    img = np.empty((size, size, 3), dtype=np.uint8)
    for i in range(3):
      img[:,:,i] = a
    return img

  orig_img = make_square_img(200)

  def run(angle):
    plt.clf()
    matplotlib.rcParams.update({'font.size': 8})
    img = ndimage.interpolation.rotate(orig_img, angle, cval=255, order=1, reshape=False)

    state = (img[:,:,0] != 255) | (img[:,:,1] != 255) | (img[:,:,2] != 255)
    plt.subplot(231)
    plt.title('State')
    plt.axis('off')
    plt.imshow(img, aspect=1)

    TSDF_TRUNC = 10.
    tsdf, sdf, depth = state_to_tsdf(state, return_all=True)
    plt.subplot(232)
    plt.axis('off')
    plt.title('Accurate')
    plt.contour(tsdf, levels=[0])
    plt.imshow(tsdf, vmin=-TSDF_TRUNC, vmax=TSDF_TRUNC, cmap='bwr')

    tsdf, sdf, _ = state_to_tsdf(state, mode='projective', return_all=True)
    plt.subplot(233)
    plt.axis('off')
    plt.title('Projective')
    plt.contour(tsdf, levels=[0])
    plt.imshow(tsdf, vmin=-TSDF_TRUNC, vmax=TSDF_TRUNC, cmap='bwr')

    mask = _state_to_visibility_mask(state)
    plt.subplot(234)
    plt.axis('off')
    plt.title('Mask')
    plt.imshow(mask)

    depth_weight = _depth_to_weights(depth)
    plt.subplot(235)
    plt.title('Depth weight')
    z = np.ones_like(mask, dtype=float)
    z *= depth_weight[:,None]
    plt.imshow(z, cmap='Greys_r')

    plt.show()

  START_ANGLE = 0
  INCR_ANGLE = 5
  for i in range(75):
    angle = START_ANGLE + i*INCR_ANGLE
    run(angle)


if __name__ == '__main__':
  # test_simple()
  # test_load_img()
  test_rotate()
