import numpy as np

# states are boolean 2d arrays. false = free space, 1 = object

TRUNC_DIST = 10.

def smooth_by_edt(phi):
  import skfmm
  d = skfmm.distance(phi)
  return np.clip(d, -TRUNC_DIST, TRUNC_DIST)

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

def state_to_tsdf(state, trunc=TRUNC_DIST, mode='accurate', return_all=False):
  assert mode in ['accurate', 'projective']

  # observe from the left
  scale = 1.

  depth = state_to_pixel_depth(state)

  if mode == 'accurate':
    # make mask where visible surface is zero
    mask = np.ones_like(state)
    for i in range(state.shape[0]):
      d = depth[i]
      if d != np.inf:
        # if the ray hits something, make a pixel zero if any neighbor is zero
        for j in range(int(d)):
          # TODO: make sure i,j are in bounds
          if state[i,j] or state[i+1,j] or state[i-1,j] or state[i,j+1] or state[i,j-1]:
            mask[i,j] = 0.
    # fill to make a SDF
    import skfmm
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

  # scale and truncate
  sdf *= scale
  tsdf = np.clip(sdf, -trunc, trunc)

  if return_all: return tsdf, sdf
  return tsdf

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
    plt.subplot(131)
    plt.title('State')
    plt.axis('off')
    plt.imshow(img, aspect=1).set_interpolation('nearest')

    TSDF_TRUNC = 10.
    tsdf, sdf = state_to_tsdf(state, return_all=True)
    plt.subplot(132)
    plt.axis('off')
    plt.title('Accurate')
    plt.contour(tsdf, levels=[0])
    plt.imshow(tsdf, vmin=-TSDF_TRUNC, vmax=TSDF_TRUNC, cmap='bwr').set_interpolation('nearest')

    tsdf, sdf = state_to_tsdf(state, mode='projective', return_all=True)
    plt.subplot(133)
    plt.axis('off')
    plt.title('Projective')
    plt.contour(tsdf, levels=[0])
    plt.imshow(tsdf, vmin=-TSDF_TRUNC, vmax=TSDF_TRUNC, cmap='bwr').set_interpolation('nearest')

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
