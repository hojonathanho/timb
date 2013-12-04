import numpy as np

# states are boolean 2d arrays. false = free space, 1 = object

def state_to_pixel_depth(state):
  depth = np.empty(state.shape[0])
  depth.fill(np.inf)
  for i in range(state.shape[0]):
    # look for first occurrence of surface
    for j in range(state.shape[1]):
      if state[i,j]:
        depth[i] = j
        break
  return depth

def state_to_tsdf(state, scale=1., trunc=10.):
  # observe from the left
  # projective point-to-point metric

  depth = state_to_pixel_depth(state)

  # fill in sdf by marching away from surface
  sdf = np.zeros_like(state, dtype=float)
  for i in range(state.shape[0]):
    d = depth[i]
    if d == np.inf:
      sdf[i,:] = np.inf
    else:
      for j in range(state.shape[1]):
        sdf[i,j] = d - j

  # scale and truncate
  return np.clip(scale*sdf, -trunc, trunc)

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
  plt.figure(2)
  plt.axis('equal')
  plt.contour(tsdf, levels=[0])
  plt.imshow(tsdf, cmap='bwr').set_interpolation('nearest')

  plt.show()

if __name__ == '__main__':
  test_simple()
  test_load_img()
