import numpy as np
import observation
import os
from scipy import ndimage
import timb


class Experiment(object):
  def set_tracker_params(self, tp):
    self.tracker_params = tp

  def get_grid_params(self):
    raise NotImplementedError

  def get_prior(self):
    return init_phi, init_weight

  def num_observations(self):
    raise NotImplementedError

  def get_state(self, i):
    raise NotImplementedError

  def get_observation(self, i):
    '''returns: tsdf, sdf, depth, weight'''
    raise NotImplementedError


class RigidRotatingSquare(Experiment):
  @staticmethod
  def _make_square_img(size, negate_inside=True):
    a = np.empty((size, size), dtype=np.uint8); a.fill(255)
    a[int(size/4.),int(size/4.):-int(size/4.)] = 0
    a[int(size/4.):-int(size/4.),int(size/4.)] = 0
    a[int(size/4.):-int(size/4.),-int(size/4.)] = 0
    a[-int(size/4.),int(size/4.):-int(size/4.)+1] = 0
    img = np.empty((size, size, 3), dtype=np.uint8)
    for i in range(3):
      img[:,:,i] = a
    return img

  def __init__(self, size=100, steps=150, start_angle=0, incr_angle=5):
    self.size, self.steps = size, steps
    self.world_min = (0., 0.)
    self.world_max = (SIZE-1., SIZE-1.)
    self.grid_params = timb.GridParams(self.world_min[0], self.world_max[0], self.world_min[1], self.world_max[1], self.size, self.size)

    self.states = []
    from scipy import ndimage
    orig_img = _make_square_img(size)
    for i in range(self.steps):
      angle = start_angle + i*incr_angle
      img = ndimage.interpolation.rotate(orig_img, angle, cval=255, order=1, reshape=False)
      self.states.append(img)

  def get_grid_params(self):
    return self.grid_params

  def get_prior(self):
    init_phi = np.ones((self.size, self.size)) * self.tracker_params.tsdf_trunc_dist
    init_weight = np.zeros((self.size, self.size))
    return init_phi, init_weight

  def num_observations(self):
    return self.steps

  def get_state(self, i):
    return self.states[i]

  def get_observation(self, i):
    return observation.observation_from_full_img(self.states[i], self.tracker_params)



def sorted_nicely(l):
  """ Sort the given iterable in the way that humans expect.""" 
  import re
  convert = lambda text: int(text) if text.isdigit() else text 
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
  return sorted(l, key = alphanum_key)


class ImageSequence(Experiment):
  @staticmethod
  def _preprocess_img(img):
    return np.transpose(img, (1, 0, 2))

  @staticmethod
  def Create(init_str):
    print init_str
    input_dir = init_str[:-2]
    use_prior_img = bool(int(init_str[-1]))
    return ImageSequence(input_dir, use_prior_img)

  def __init__(self, input_dir, use_prior_img):
    files = [(input_dir + '/' + f) for f in os.listdir(input_dir) if os.path.isfile(input_dir + '/' + f)]
    imgs = [self._preprocess_img(ndimage.imread(f)) for f in sorted_nicely(files)]
    self.states = [((img[:,:,0] != 255) | (img[:,:,1] != 255) | (img[:,:,2] != 255)) for img in imgs]

    self.prior_img = None
    if use_prior_img:
      print 'Using prior image'
      self.prior_img = self._preprocess_img(ndimage.imread(os.path.join(input_dir, 'prior.png')))

    self.size, self.steps = 100, len(self.states)
    self.world_min = (0., 0.)
    self.world_max = (self.size-1., self.size-1.)
    self.grid_params = timb.GridParams(self.world_min[0], self.world_max[0], self.world_min[1], self.world_max[1], self.size, self.size)

  def get_grid_params(self):
    return self.grid_params

  def get_prior(self):
    init_phi = np.ones((self.size, self.size)) * self.tracker_params.tsdf_trunc_dist
    init_weight = np.zeros((self.size, self.size))

    if self.prior_img is not None:
      state = (self.prior_img[:,:,0] == 0) & (self.prior_img[:,:,1] == 0) & (self.prior_img[:,:,2] == 0)
      sdf = ndimage.morphology.distance_transform_edt(~state)
      sdf[self.prior_img[:,:,0] != 255] *= -1.
      init_phi = np.clip(sdf, -self.tracker_params.tsdf_trunc_dist, self.tracker_params.tsdf_trunc_dist)
      init_weight.fill(1)

    return init_phi, init_weight

  def num_observations(self):
    return self.steps

  def get_state(self, i):
    return self.states[i]

  def get_observation(self, i):
    return observation.observation_from_full_state(self.states[i], self.tracker_params)


def run_experiment(ex, tracker_params, callback=None, iter_cap=None):
  ex.set_tracker_params(tracker_params)
  grid_params = ex.get_grid_params()

  curr_phi, curr_weight = ex.get_prior()

  experiment_log = []

  num = ex.num_observations() if iter_cap is None else min(ex.num_observations(), iter_cap)
  for i in range(num):
    iter_data = {}

    iter_data['curr_phi'], iter_data['curr_weight'] = curr_phi, curr_weight

    obs_tsdf, obs_sdf, obs_depth, obs_weight, obs_trust_mask = ex.get_observation(i)
    iter_data['state'] = ex.get_state(i)
    iter_data['obs_tsdf'], iter_data['obs_sdf'], iter_data['obs_depth'], iter_data['obs_weight'], iter_data['obs_trust_mask'] = obs_tsdf, obs_sdf, obs_depth, obs_weight, obs_trust_mask

    new_phi, new_weight, problem_data = timb.run_one_step(
      grid_params, tracker_params,
      obs_tsdf, obs_weight, obs_trust_mask,
      curr_phi, curr_weight,
      return_full=True
    )
    iter_data['new_phi'], iter_data['new_weight'], iter_data['problem_data'] = new_phi, new_weight, problem_data

    iter_data['trusted'] = trusted = timb.threshold_trusted(tracker_params, new_phi, new_weight)
    iter_data['output'] = np.where(trusted, new_phi, np.nan)

    if callback is not None:
      callback(i, iter_data)

    experiment_log.append(iter_data)
    curr_phi, curr_weight = new_phi, new_weight

  return experiment_log
