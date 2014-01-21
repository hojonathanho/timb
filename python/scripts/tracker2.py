import numpy as np
import interpolation as interp
import timb
import observation
from scipy import ndimage

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default=None)
parser.add_argument('--dump_dir', default=None)
parser.add_argument('--input_dir', default=None)
parser.add_argument('--do_not_smooth', action='store_true')
args = parser.parse_args()

np.set_printoptions(linewidth=1000)

def print_result(r):
  print 'phi'
  print r.phi
  print 'u_x'
  print r.u_x
  print 'u_y'
  print r.u_y
  # print 'next_phi'
  # print r.next_phi
  # print 'next_omega'
  # print r.next_omega

def test1():
  SIZE = 5
  WORLD_MIN = (0., 0.)
  WORLD_MAX = (SIZE-1., SIZE-1.)
  gp = timb.GridParams(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], SIZE, SIZE)

  # initial state: zero precision
  init_phi = np.empty((SIZE, SIZE)); init_phi.fill(-1.)
  init_omega = np.zeros((SIZE, SIZE))
  obs_mask = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0]
  ]).astype(float)
  obs_vals = np.array([
    [2, 1, 0, 0, 0],
    [2, 1, 0, 0, 0],
    [2, 1, 0, 0, 0],
    [2, 1, 0, 0, 0],
    [2, 1, 0, 0, 0]
  ]).astype(float)

  tracker = timb.Tracker(gp)
  tracker.opt.params().check_linearizations = True
  tracker.observation_cost.set_observation(obs_vals, obs_mask)
  tracker.agreement_cost.set_prev_phi_and_weights(init_phi, init_omega)

  # initialization: previous phi, zero flow
  init_state = timb.State(init_phi, np.zeros_like(init_phi), np.zeros_like(init_phi))
  result = tracker.optimize(init_state)
  print_result(result)

  desired = np.array([
    [2, 1, -1, -1, -1],
    [2, 1, -1, -1, -1],
    [2, 1, -1, -1, -1],
    [2, 1, -1, -1, -1],
    [2, 1, -1, -1, -1]
  ])
  assert abs(desired - result.phi).max() < 1e-3

def test2():
  SIZE = 5
  WORLD_MIN = (0., 0.)
  WORLD_MAX = (SIZE-1., SIZE-1.)

  # initial state: zero precision
  init_phi = np.array([
    [0, 2, 1, 0, 0],
    [0, 2, 1, 0, 0],
    [0, 2, 1, 0, 0],
    [0, 2, 1, 0, 0],
    [0, 2, 1, 0, 0]
  ]).astype(float)
  init_omega = np.zeros((SIZE, SIZE))
  obs_mask = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0]
  ]).astype(float)
  obs_vals = np.array([
    [2, 1, 0, 0, 0],
    [2, 1, 0, 0, 0],
    [2, 1, 0, 0, 0],
    [2, 1, 0, 0, 0],
    [2, 1, 0, 0, 0]
  ]).astype(float)

  prob = ctimbpy.TrackingProblem(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], SIZE, SIZE)
  prob.set_obs(obs_vals, obs_mask)
  prob.set_prior(init_phi, init_omega)

  result = prob.optimize()
  print_result(result)

  desired = np.array([
    [2, 1, -1, -1, -1],
    [2, 1, -1, -1, -1],
    [2, 1, -1, -1, -1],
    [2, 1, -1, -1, -1],
    [2, 1, -1, -1, -1]
  ])
  assert abs(desired - result.next_phi).max() < 1e-3


def test_should_move():
  SIZE = 5
  WORLD_MIN = (0., 0.)
  WORLD_MAX = (SIZE-1., SIZE-1.)
  gp = timb.GridParams(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], SIZE, SIZE)

  # initial state: zero precision
  init_phi = np.array([
    [0, 2, 1, 0, 0],
    [0, 2, 1, 0, 0],
    [0, 2, 1, 0, 0],
    [0, 2, 1, 0, 0],
    [0, 2, 1, 0, 0]
  ]).astype(float)
  init_omega = np.zeros((SIZE, SIZE)); init_omega.fill(1.)
  obs_mask = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0]
  ]).astype(float)
  obs_vals = np.array([
    [2, 1, 0, 0, 0],
    [2, 1, 0, 0, 0],
    [2, 1, 0, 0, 0],
    [2, 1, 0, 0, 0],
    [2, 1, 0, 0, 0]
  ]).astype(float)

  tracker = timb.Tracker(gp)
  tracker.opt.params().check_linearizations = True
  tracker.opt.params().keep_results_over_iterations = True
  tracker.observation_cost.set_observation(obs_vals, obs_mask)
  tracker.agreement_cost.set_prev_phi_and_weights(init_phi, init_omega)
  # initialization: previous phi, zero flow
  init_state = timb.State(gp, init_phi, np.zeros_like(init_phi), np.zeros_like(init_phi))
  result, opt_result = tracker.optimize(init_state)
  print_result(result)
  timb.plot_state(result)

  for i, i_x in enumerate(opt_result.x_over_iters):
    print i
    s = timb.State.FromPacked(gp, i_x)
    print_result(s)
    timb.plot_state(s)


def make_square_img(SIZE, negate_inside=True):
  a = np.empty((SIZE, SIZE), dtype=np.uint8); a.fill(255)
  a[int(SIZE/4.),int(SIZE/4.):-int(SIZE/4.)] = 0
  a[int(SIZE/4.):-int(SIZE/4.),int(SIZE/4.)] = 0
  a[int(SIZE/4.):-int(SIZE/4.),-int(SIZE/4.)] = 0
  a[-int(SIZE/4.),int(SIZE/4.):-int(SIZE/4.)+1] = 0
  img = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
  for i in range(3):
    img[:,:,i] = a
  return img




def rot(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s], [s, c]], dtype=float)
def generate_rot_flow(size, angle):
  u = np.zeros((size, size, 2))
  c = np.array([size/2., size/2.])
  R = rot(angle)
  for i in range(size):
    for j in range(size):
      x = np.array([i,j], dtype=float)
      d = x - c
      u[i,j,:] = R.dot(d) - d
  return u
# SIZE = 100
# WORLD_MIN = (0., 0.)
# WORLD_MAX = (SIZE-1., SIZE-1.)
# import matplotlib.pyplot as plt
# def plot_u(u):
#   x = np.linspace(WORLD_MIN[0], WORLD_MAX[0], u.shape[0])
#   y = np.linspace(WORLD_MIN[1], WORLD_MAX[1], u.shape[1])
#   Y, X = np.meshgrid(x, y)
#   plt.axis('equal')
#   plt.quiver(X, Y, u[:,:,0], u[:,:,1], angles='xy', scale_units='xy', scale=1.)
# plot_u(generate_rot_flow(SIZE, 5*np.pi/180))
# plt.show()

def test_image():
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  TSDF_TRUNC = observation.TRUNC_DIST

  SIZE = 100
  FIRST_OBS_EXTRA_WEIGHT = 2
  START_ANGLE = 0
  INCR_ANGLE = 5

  # SIZE = tsdf.shape[0]
  WORLD_MIN = (0., 0.)
  WORLD_MAX = (SIZE-1., SIZE-1.)
  gp = timb.GridParams(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], SIZE, SIZE)

  def run(obs_num, img, init_phi, init_weight):
    problem_data = {}

    state = (img[:,:,0] != 255) | (img[:,:,1] != 255) | (img[:,:,2] != 255)
    problem_data['state'] = state
    problem_data['init_phi'] = init_phi
    problem_data['init_weight'] = init_weight

    tsdf, sdf, depth = observation.state_to_tsdf(state, return_all=True)
    problem_data['obs_tsdf'], problem_data['obs_sdf'], problem_data['obs_depth'] = tsdf, sdf, depth

    obs_weight = observation.compute_obs_weight(sdf, depth)
    problem_data['obs_weight'] = obs_weight

    timb.Coeffs.flow_norm = 1e-6
    timb.Coeffs.flow_rigidity = 1e-1
    timb.Coeffs.observation = 1.
    timb.Coeffs.prior = 1.
    tracker = timb.Tracker(gp)
    tracker.opt.params().check_linearizations = False
    tracker.opt.params().keep_results_over_iterations = False
    tracker.opt.params().max_iter = 5
    tracker.opt.params().approx_improve_rel_tol = 1e-8
    tracker.set_observation(tsdf, obs_weight)
    tracker.set_prev_phi_and_weights(init_phi, init_weight)
    # initialization: previous phi, zero flow
    init_u = np.zeros(init_phi.shape + (2,))
    # if obs_num != 0: init_u = generate_rot_flow(SIZE, INCR_ANGLE*np.pi/180) # optimization starting point
    result, opt_result = tracker.optimize(timb.State(gp, init_phi, init_u[:,:,0], init_u[:,:,1]))

    problem_data['cost_over_iters'] = opt_result.cost_over_iters

    problem_data['u_x'], problem_data['u_y'] = result.u_x, result.u_y

    problem_data['new_phi'] = result.phi

    flowed_init_weight = timb.apply_flow(gp, init_weight, result.u_x, result.u_y)
    next_weight = flowed_init_weight + obs_weight
    problem_data['new_weight'] = next_weight

    if args.do_not_smooth:
      next_phi = result.phi
    else:
      smoother_ignore_region = (abs(result.phi) > TSDF_TRUNC/2) | (abs(next_weight) < 1e-2)
      smoother_weights = np.where(smoother_ignore_region, 0, next_weight)
      next_phi = timb.smooth(result.phi, smoother_weights)
    next_phi = np.clip(next_phi, -TSDF_TRUNC, TSDF_TRUNC)
    problem_data['new_phi_smoothed'] = next_phi

    timb.plot_problem_data(plt, TSDF_TRUNC, gp, state, tsdf, obs_weight, init_phi, init_weight, result, opt_result, next_phi, next_weight)

    if args.output_dir is None:
      plt.show()
    else:
      plt.savefig('%s/plots_%d.png' % (args.output_dir, obs_num), bbox_inches='tight')

    if args.dump_dir is not None:
      import cPickle
      path = '%s/dump_%d.pkl' % (args.dump_dir, obs_num)
      with open(path, 'w') as f:
        cPickle.dump(problem_data, f)
      print 'wrote to', path

    return next_phi, next_weight


  orig_phi = np.empty((SIZE, SIZE)); orig_phi.fill(TSDF_TRUNC)
  orig_omega = np.zeros((SIZE, SIZE));# orig_omega.fill(.0001)

  curr_phi, curr_omega = orig_phi, orig_omega

  if args.input_dir is not None:

    def sorted_nicely(l):
      """ Sort the given iterable in the way that humans expect.""" 
      import re
      convert = lambda text: int(text) if text.isdigit() else text 
      alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
      return sorted(l, key = alphanum_key)

    import os
    files = [(args.input_dir + '/' + f) for f in os.listdir(args.input_dir) if os.path.isfile(args.input_dir + '/' + f)]
    images = [ndimage.imread(f) for f in sorted_nicely(files)]
    import matplotlib; import matplotlib.pyplot as plt
    for i, img in enumerate(images):
      img = np.transpose(img, (1, 0, 2))
      # state = (img[:,:,0] != 255) | (img[:,:,1] != 255) | (img[:,:,2] != 255)
      # plt.imshow(state.T, origin='lower')
      # plt.show()
      curr_phi, curr_omega = run(i, img, curr_phi, curr_omega)

  else:
    orig_img = make_square_img(SIZE)
    for i in range(150):
      angle = START_ANGLE + i*INCR_ANGLE
      img = ndimage.interpolation.rotate(orig_img, angle, cval=255, order=1, reshape=False)
      curr_phi, curr_omega = run(i, img, curr_phi, curr_omega)

if __name__ == '__main__':
  test_image()
