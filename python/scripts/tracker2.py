import numpy as np
import interpolation as interp
import ctimbpy

np.set_printoptions(linewidth=1000)

SIZE = 5
WORLD_MIN = (0., 0.)
WORLD_MAX = (SIZE-1., SIZE-1.)

def test1():
  prob = ctimbpy.TrackingProblem(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], SIZE, SIZE)

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

  prob.set_obs(obs_vals, obs_mask)
  prob.set_prior(init_phi, init_omega)

  result = prob.optimize()
  print 'phi'
  print result.phi
  print 'u'
  print result.u
  print 'next_phi'
  print result.next_phi
  print 'next_omega'
  print result.next_omega

if __name__ == '__main__':
  test1()
