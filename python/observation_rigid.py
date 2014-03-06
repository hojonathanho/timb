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


def state_to_tsdf(state, trunc_dist, mode='accurate', return_all=False):
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


def compute_obs_weight_rigid(obs_sdf, depth, epsilon, delta):
  # linear weight (b) in Bylow et al 2013
  # epsilon, delta = 5, 10
  # epsilon, delta = 0, 5
  assert delta > epsilon
  w = np.where(obs_sdf >= -epsilon, OBS_PEAK_WEIGHT, obs_sdf)
  w = np.where((obs_sdf <= -epsilon) & (obs_sdf >= -delta), (OBS_PEAK_WEIGHT/(delta-epsilon))*(obs_sdf + delta), w)
  w = np.where(obs_sdf < -delta, 0, w)
  w = np.where(np.isfinite(obs_sdf), w, 0) # zero precision where we get inf/no depth measurement
  return w


def observation_from_full_state_rigid(state, tracker_params):
  tsdf, sdf, depth = state_to_tsdf(state, trunc_dist=tracker_params.tsdf_trunc_dist,
                                   mode=tracker_params.sensor_mode,
                                   return_all=True)

  weight = compute_obs_weight_rigid( sdf, depth,
                                     epsilon=tracker_params.obs_weight_epsilon,
                                     delta=tracker_params.obs_weight_delta  )
  return tsdf, sdf, depth, weight
