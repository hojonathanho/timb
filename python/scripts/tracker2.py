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


def compute_obs_weight(obs_tsdf, weight_max):
  # a = abs(obs_tsdf)
  # return np.where(a < trunc_val, (weight_max/trunc_val)*(trunc_val-a), 0.)

  # linear weight (b) in Bylow et al 2013
  epsilon, delta = 5, 10
  assert delta > epsilon
  w = np.where(obs_tsdf >= -epsilon, weight_max, obs_tsdf)
  w = np.where((obs_tsdf <= -epsilon) & (obs_tsdf >= -delta), (weight_max/(delta-epsilon))*(obs_tsdf + delta), w)
  w = np.where(obs_tsdf < -delta, 0, w)
  return w

def test_image():
  import matplotlib.pyplot as plt
  from scipy import ndimage
  import observation

  TSDF_TRUNC = 10
  OBS_PEAK_WEIGHT = 100.

  def run(angle, init_phi, init_omega):
    img = ndimage.interpolation.rotate(orig_img, angle, cval=255, order=1, reshape=False)

    plt.imshow(img).set_interpolation('nearest')

    state = (img[:,:,0] != 255) | (img[:,:,1] != 255) | (img[:,:,2] != 255)
    plt.figure(1)
    plt.contourf(state)

    tsdf, sdf = observation.state_to_tsdf(state, return_all=True)
    print np.count_nonzero(abs(tsdf) < TSDF_TRUNC), tsdf.size
    plt.figure(2)
    plt.axis('equal')
    plt.contour(tsdf, levels=[0])
    plt.imshow(tsdf, cmap='bwr').set_interpolation('nearest')

    SIZE = tsdf.shape[0]
    WORLD_MIN = (0., 0.)
    WORLD_MAX = (SIZE-1., SIZE-1.)
    prob = ctimbpy.TrackingProblem(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], SIZE, SIZE)
    prob.coeffs.flow_norm = 1e-9
    prob.coeffs.flow_rigidity = 1
    prob.coeffs.observation = 1.
    prob.coeffs.prior = 1.
    prob.set_obs(tsdf, abs(tsdf) < TSDF_TRUNC)
    prob.set_prior(init_phi, init_omega)
    result = prob.optimize()
    plt.figure(3)
    plt.subplot(121)
    plt.imshow(result.phi, cmap='bwr').set_interpolation('nearest')
    plt.subplot(122)
    plt.imshow(result.next_phi, cmap='bwr').set_interpolation('nearest')

    plt.figure(4)
    omega = compute_obs_weight(sdf, OBS_PEAK_WEIGHT)
    plt.imshow(omega, cmap='binary', vmin=0, vmax=OBS_PEAK_WEIGHT*10).set_interpolation('nearest')

    next_omega = result.next_omega + omega

    plt.show()

    return result.next_phi, next_omega


  orig_img = ndimage.imread('/Users/jonathan/code/timb/data/smallbear2.png')
  SIZE = orig_img.shape[0]

  orig_phi = np.empty((SIZE, SIZE)); orig_phi.fill(TSDF_TRUNC)
  orig_omega = np.zeros((SIZE, SIZE))
  
  start_angle = 90
  curr_phi, curr_omega = orig_phi, orig_omega
  for i in range(100):
    angle = start_angle + i*5
    curr_phi, curr_omega = run(angle, curr_phi, curr_omega)

if __name__ == '__main__':
  test_image()
