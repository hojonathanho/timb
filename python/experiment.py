import numpy as np
import observation


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
    return observation.observation_from_full_img(self.states[i], tracker_params)


def run_experiment(ex, tracker_params, callback=None):
  ex.set_tracker_params(tracker_params)
  grid_params = ex.get_grid_params()

  curr_phi, curr_weight = ex.get_prior()

  experiment_log = []

  for i in range(ex.num_observations()):
    iter_data = {}

    iter_data['curr_phi'], iter_data['curr_weight'] = curr_phi, curr_weight

    obs_tsdf, obs_sdf, obs_depth, obs_weight = ex.get_observation(i)
    iter_data['obs_tsdf'], iter_data['obs_sdf'], iter_data['obs_depth'], iter_data['obs_weight'] = obs_tsdf, obs_sdf, obs_depth, obs_weight

    new_phi, new_weight, problem_data = timb.run_one_step(
      grid_params, tracker_params,
      obs_tsdf, obs_weight,
      curr_phi, curr_weight,
      return_full=True
    )
    iter_data['new_phi'], iter_data['new_weight'], iter_data['problem_data'] = new_phi, new_weight, problem_data

    if callback is not None:
      callback(i, iter_data)

    experiment_log.append(iter_data)
    curr_phi, curr_weight = new_phi, new_weight

  return experiment_log
