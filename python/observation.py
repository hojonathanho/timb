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
      for j in range(int(d) + 1):
        try:
          if state[i,j] or state[i+1,j] or state[i-1,j]:
            mask[i,j] = 0
        except IndexError:
          pass
  return mask


def _find_first_inf(depth):
  first = 0
  for i in range(len(depth)):
    if depth[i] != np.inf:
      first = i
      break
  return first

def _find_last_inf(depth):
  last = len(depth) - 1
  for i in range(len(depth)):
    if depth[len(depth)-i-1] != np.inf:
      last = len(depth)-i-1
      break
  return last

def _make_edge_downweight(length, downweight_size, downweight_front=False, downweight_back=False):
  assert downweight_front or downweight_back

  w = np.ones(length)
  m = min(downweight_size, length)
  if downweight_front:
    w[:m] = np.minimum(w[:m], np.linspace(0, 1, m))
  if downweight_back:
    w[-m:] = np.minimum(w[-m:], np.linspace(1, 0, m))
  return w

def _depth_to_weights(depth, trunc_dist, filter_radius, use_linear_downweight, use_min_to_combine):
  '''
  Computes weights for a depth image
  '''

  # Downweight around regions with no measurements
  has_measurement = (depth != np.inf).astype(float)

  filt = np.ones(2*filter_radius + 1); filt /= filt.sum()
  out = np.convolve(has_measurement, filt, 'same')
  def f(x):
    '''maps [.5, 1] to [0, 1], differentiable at boundary'''
    y = .5*np.sin(2*np.pi*(x - .75)) + .5
    y[x < .5] = 0.
    y[x > 1.] = 1.
    return y
  def g(x):
    '''maps [.5, 1] to [0, 1] in a linear way'''
    return np.clip(2.*(x - .5), 0., 1.)

  if use_linear_downweight:
    w = g(out)
  else:
    w = f(out)

  first, last = _find_first_inf(depth), _find_last_inf(depth)

  # Downweight around depth discontinuities
  discont_radius = int(filter_radius/2.) # FIXME: ARBITRARY
  w2 = np.ones_like(w)
  grad = np.convolve(depth[first:last+1], [.5, -.5], 'same')[1:-1]
  decay = np.r_[np.linspace(1, 0, discont_radius), 0, np.linspace(0, 1, discont_radius)]
  for i in range(len(grad)):
    if abs(grad[i]) > 3:
      offset = first + i + 1
      if use_min_to_combine:
        w2[offset-discont_radius:offset+discont_radius+1] = np.minimum(w2[offset-discont_radius:offset+discont_radius+1], decay)
      else:
        w2[offset-discont_radius:offset+discont_radius+1] *= decay
  w = np.minimum(w, w2)

  return w


def state_to_tsdf(state, trunc_dist=10, mode='accurate', return_all=False):
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
        # sdf[i,d+1+trunc_dist:] = -trunc_dist

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


def compute_obs_weight(obs_sdf, depth, trunc_dist, epsilon, delta, filter_radius, use_linear_downweight, use_min_to_combine, weight_far):
  # linear weight (b) in Bylow et al 2013
  # epsilon, delta = 5, 10
  # epsilon, delta = 0, 5
  assert delta > epsilon
  w = np.where(obs_sdf >= -epsilon, OBS_PEAK_WEIGHT, obs_sdf)
  w = np.where((obs_sdf <= -epsilon) & (obs_sdf >= -delta), (OBS_PEAK_WEIGHT/(delta-epsilon))*(obs_sdf + delta), w)
  w = np.where(obs_sdf < -delta, 0, w)
  w = np.where(np.isfinite(obs_sdf), w, 0) # zero precision where we get inf/no depth measurement

  dw = _depth_to_weights(depth, trunc_dist, filter_radius, use_linear_downweight, use_min_to_combine)
  w *= dw[:,None]

  # Set weight to 1 everywhere TSDF_TRUNC away from the object
  always_trust_mask = np.zeros_like(w, dtype=bool)
  if weight_far:
    first, last = _find_first_inf(depth), _find_last_inf(depth)

    w[:max(0, first-trunc_dist),:] = _make_edge_downweight(max(0, first-trunc_dist), filter_radius, downweight_back=True)[:,None]
    always_trust_mask[:max(0, first-trunc_dist),:] = True

    w[min(last+trunc_dist, len(depth)-1):,:] = _make_edge_downweight(len(depth)-min(last+trunc_dist, len(depth)-1), filter_radius, downweight_front=True)[:,None]
    always_trust_mask[min(last+trunc_dist, len(depth)-1):,:] = True

  return w, always_trust_mask


def observation_from_full_state(state, tracker_params):
  tsdf, sdf, depth = state_to_tsdf(
    state,
    trunc_dist=tracker_params.tsdf_trunc_dist,
    mode=tracker_params.sensor_mode,
    return_all=True
  )

  weight, always_trust_mask = compute_obs_weight(
    sdf,
    depth,
    tracker_params.tsdf_trunc_dist,
    epsilon=tracker_params.obs_weight_epsilon,
    delta=tracker_params.obs_weight_delta,
    filter_radius=tracker_params.obs_weight_filter_radius,
    use_linear_downweight=tracker_params.use_linear_downweight,
    use_min_to_combine=tracker_params.use_min_to_combine,
    weight_far=tracker_params.obs_weight_far
  )

  return tsdf, sdf, depth, weight, always_trust_mask


def observation_from_full_img(img, tracker_params):
  state = (img[:,:,0] != 255) | (img[:,:,1] != 255) | (img[:,:,2] != 255)
  return observation_from_full_state(state, tracker_params)


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
  plt.contourf(tsdf, cmap='bwr')
  print tsdf
  plt.show()

  assert np.allclose(tsdf, np.array(
    [[ 10.,  10.,  10.,  10.,  10.],
     [  1.,   0.,  -1.,  -2.,  -3.],
     [  1.,   0.,  -1.,  -2.,  -3.],
     [  1.,   0.,  -1.,  -2.,  -3.],
     [ 10.,  10.,  10.,  10.,  10.]]
  ))


def test_load_img(img_fname):
  import matplotlib.pyplot as plt
  from scipy import ndimage

  img = ndimage.imread(img_fname)

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

  def run(img):
    plt.clf()
    matplotlib.rcParams.update({'font.size': 8})

    state = (img[:,:,0] != 255) | (img[:,:,1] != 255) | (img[:,:,2] != 255)
    plt.subplot(331)
    plt.title('State')
    plt.axis('off')
    plt.imshow(img, aspect=1)

    TSDF_TRUNC = 10.
    tsdf, sdf, depth = state_to_tsdf(state, TSDF_TRUNC, return_all=True)
    plt.subplot(332)
    plt.axis('off')
    plt.title('Accurate')
    plt.contour(tsdf, levels=[0])
    plt.imshow(tsdf, vmin=-TSDF_TRUNC, vmax=TSDF_TRUNC, cmap='bwr')

    tsdf, sdf, _ = state_to_tsdf(state, TSDF_TRUNC, mode='projective', return_all=True)
    plt.subplot(333)
    plt.axis('off')
    plt.title('Projective')
    plt.contour(tsdf, levels=[0])
    plt.imshow(tsdf, vmin=-TSDF_TRUNC, vmax=TSDF_TRUNC, cmap='bwr')

    mask = _state_to_visibility_mask(state)
    plt.subplot(334)
    plt.axis('off')
    plt.title('Mask')
    plt.imshow(mask)

    depth_weight = _depth_to_weights(depth, TSDF_TRUNC, 20, use_linear_downweight=True, use_min_to_combine=True)
    #weight, ignore = compute_obs_weight(sdf, depth, TSDF_TRUNC, 0, 5, 20)
    plt.subplot(335)
    plt.title('Depth weight')
    z = np.ones_like(mask, dtype=float)
    z *= depth_weight[:,None]
    plt.imshow(z, cmap='Greys_r')
    plt.subplot(336)
    #plt.plot(weight[:,50])
    plt.plot(z)

    plt.subplot(337)
    depth_weight = _depth_to_weights(depth, TSDF_TRUNC, 20, use_linear_downweight=False, use_min_to_combine=True)
    z = np.ones_like(mask, dtype=float)
    z *= depth_weight[:,None]
    plt.plot(z)

    plt.show()

  def gen_img_sequence():
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

    START_ANGLE = 0
    INCR_ANGLE = 5
    orig_img = make_square_img(200)
    for i in range(75):
      angle = START_ANGLE + i*INCR_ANGLE
      img = ndimage.interpolation.rotate(orig_img, angle, cval=255, order=1, reshape=False)
      yield img


  def gen_img_sequence2():
    import os
    input_dir = os.path.expanduser('~/Dropbox/research/tracking/data/in/simple_bend')
    files = [(input_dir + '/' + f) for f in os.listdir(input_dir) if os.path.isfile(input_dir + '/' + f)]
    for f in sorted(files):
      print f
      yield np.transpose(ndimage.imread(f), (1, 0, 2))

  i = 0
  for img in gen_img_sequence2():
    i += 1
    if i < 10: continue
    run(img)


if __name__ == '__main__':
  #test_simple()
  test_load_img('/home/ankush/sandbox444/timb/data/smallbear2.png')
  #test_rotate()
