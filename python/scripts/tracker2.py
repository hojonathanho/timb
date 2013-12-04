import numpy as np
import interpolation as interp
import ctimbpy

np.set_printoptions(linewidth=1000)

def print_result(r):
  print 'phi'
  print r.phi
  print 'u'
  print r.u
  print 'next_phi'
  print r.next_phi
  print 'next_omega'
  print r.next_omega

def test1():
  SIZE = 5
  WORLD_MIN = (0., 0.)
  WORLD_MAX = (SIZE-1., SIZE-1.)

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

  # desired = np.array([
  #   [2, 1, -1, -1, -1],
  #   [2, 1, -1, -1, -1],
  #   [2, 1, -1, -1, -1],
  #   [2, 1, -1, -1, -1],
  #   [2, 1, -1, -1, -1]
  # ])
  # assert abs(desired - result.next_phi).max() < 1e-3


def test_should_move():
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
  init_omega = np.zeros((SIZE, SIZE)); init_omega.fill(100)
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

def compute_obs_weight(obs_sdf, weight_max):
  # a = abs(obs_tsdf)
  # return np.where(a < trunc_val, (weight_max/trunc_val)*(trunc_val-a), 0.)

  # linear weight (b) in Bylow et al 2013
  epsilon, delta = 5, 10
  assert delta > epsilon
  w = np.where(obs_sdf >= -epsilon, weight_max, obs_sdf)
  w = np.where((obs_sdf <= -epsilon) & (obs_sdf >= -delta), (weight_max/(delta-epsilon))*(obs_sdf + delta), w)
  w = np.where(obs_sdf < -delta, 0, w)
  w = np.where(np.isfinite(obs_sdf), w, 0) # zero precision where we get inf/no depth measurement
  return w


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
  import matplotlib.pyplot as plt
  from scipy import ndimage
  import observation

  TSDF_TRUNC = 10
  OBS_PEAK_WEIGHT = 1.

  # orig_img = ndimage.imread('/Users/jonathan/code/timb/data/smallbear2.png')
  orig_img = make_square_img(100)
  SIZE = orig_img.shape[0]
  FIRST_OBS_EXTRA_WEIGHT = 20
  START_ANGLE = 0
  INCR_ANGLE = 5

  def run(angle, init_phi, init_omega):
    plt.clf()
    matplotlib.rcParams.update({'font.size': 8})
    img = ndimage.interpolation.rotate(orig_img, angle, cval=255, order=1, reshape=False)

    state = (img[:,:,0] != 255) | (img[:,:,1] != 255) | (img[:,:,2] != 255)
    plt.subplot(251)
    plt.title('State')
    plt.axis('off')
    plt.imshow(img, aspect=1).set_interpolation('nearest')

    tsdf, sdf = observation.state_to_tsdf(state, return_all=True)
    # print np.count_nonzero(abs(tsdf) < TSDF_TRUNC), tsdf.size
    plt.subplot(252)
    plt.axis('off')
    plt.title('Observation TSDF')
    plt.contour(tsdf, levels=[0])
    plt.imshow(tsdf, vmin=-TSDF_TRUNC, vmax=TSDF_TRUNC, cmap='bwr').set_interpolation('nearest')

    plt.subplot(253)
    plt.title('Observation weight')
    plt.axis('off')
    obs_weights = compute_obs_weight(sdf, OBS_PEAK_WEIGHT)
    if angle == 0: obs_weights *= FIRST_OBS_EXTRA_WEIGHT
    plt.imshow(obs_weights, cmap='binary', vmin=0, vmax=OBS_PEAK_WEIGHT*10).set_interpolation('nearest')

    plt.subplot(254)
    plt.title('Prior TSDF')
    plt.axis('off')
    plt.imshow(init_phi, vmin=-TSDF_TRUNC, vmax=TSDF_TRUNC, cmap='bwr').set_interpolation('nearest')
    plt.subplot(255)
    plt.title('Prior weight')
    plt.axis('off')
    plt.imshow(init_omega, vmin=0, vmax=OBS_PEAK_WEIGHT*10, cmap='binary').set_interpolation('nearest')

    SIZE = tsdf.shape[0]
    WORLD_MIN = (0., 0.)
    WORLD_MAX = (SIZE-1., SIZE-1.)
    prob = ctimbpy.TrackingProblem(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], SIZE, SIZE)
    prob.coeffs.flow_norm = 1e-9
    prob.coeffs.flow_rigidity = 1
    prob.coeffs.observation = 1.
    prob.coeffs.prior = 1.
    prob.set_obs(tsdf, obs_weights, np.ones_like(tsdf))#abs(tsdf) < TSDF_TRUNC)
    prob.set_prior(init_phi, init_omega)

    if angle != 0: prob.set_init_u(generate_rot_flow(SIZE, INCR_ANGLE*np.pi/180)) # optimization starting point
    result = prob.optimize()

    plt.subplot(256)
    plt.title('TSDF')
    plt.axis('off')
    plt.imshow(result.phi, vmin=-TSDF_TRUNC, vmax=TSDF_TRUNC, cmap='bwr').set_interpolation('nearest')

    plt.subplot(257, aspect='equal')
    plt.title('Flow')
    def plot_u(u):
      x = np.linspace(WORLD_MIN[0], WORLD_MAX[0], u.shape[0])
      y = np.linspace(WORLD_MIN[1], WORLD_MAX[1], u.shape[1])
      Y, X = np.meshgrid(x, y)
      plt.quiver(X, Y, u[:,:,0], u[:,:,1], angles='xy', scale_units='xy', scale=1.)
    plot_u(result.u)

    plt.subplot(258)
    plt.title('New TSDF')
    plt.axis('off')
    plt.imshow(result.next_phi, vmin=-TSDF_TRUNC, vmax=TSDF_TRUNC, cmap='bwr').set_interpolation('nearest')

    next_omega = result.next_omega + obs_weights
    plt.subplot(259)
    plt.title('New weight')
    plt.axis('off')
    plt.imshow(next_omega, vmin=0, vmax=OBS_PEAK_WEIGHT*10, cmap='binary').set_interpolation('nearest')

    plt.subplot(2,5,10)
    plt.title('Output')
    plt.axis('off')
    plt.contour(np.where(next_omega > .5, result.next_phi, np.nan), levels=[0])
    # plt.imshow(next_omega > .5, vmin=0, vmax=1, cmap='binary').set_interpolation('nearest')
    plt.imshow(np.zeros_like(tsdf), vmin=0, vmax=1, cmap='binary').set_interpolation('nearest')

    # plt.show()
    plt.savefig('out/plots_%d.png' % angle, bbox_inches='tight')

    return result.next_phi, next_omega


  orig_phi = np.empty((SIZE, SIZE)); orig_phi.fill(TSDF_TRUNC)
  orig_omega = np.zeros((SIZE, SIZE))
  
  curr_phi, curr_omega = orig_phi, orig_omega
  for i in range(75):
    angle = START_ANGLE + i*INCR_ANGLE
    curr_phi, curr_omega = run(angle, curr_phi, curr_omega)

if __name__ == '__main__':
  test_image()
