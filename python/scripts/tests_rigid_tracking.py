import numpy as np
import rigid_tracker as rt
import ctimb
import observation
from scipy import ndimage
import time
import argparse
import os.path as osp

TEST_DIR = '/home/ankush/sandbox444/timb/data/test_cases'

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default=None)
parser.add_argument('--dump_dir', default=None)
parser.add_argument('--test_case_name', default=None)
parser.add_argument('--do_not_smooth', action='store_true')
parser.add_argument('--prior_img', default=None, type=str)
args = parser.parse_args()

if args.test_case_name:
  args.input_dir = osp.join(TEST_DIR, args.test_case_name)
else:
  args.input_dir = None

np.set_printoptions(linewidth=1000)


def make_square_img(SIZE):
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

def pad_state(state, npad):
  """
  state : a 2D matrix.
  npad  : the number of additional pixels to be added on each side.
  """
  l,w = state.shape
  pad_state = np.zeros(np.array([l,w]) + 2*npad, dtype=state.dtype)
  pad_state[npad:npad+l, npad:npad+w] = state
  return pad_state


def test_image():
  import matplotlib
  if args.output_dir is not None: matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  NPAD = 50
  SIZE = 100
  PSIZE = SIZE+2*NPAD
  START_ANGLE = 0
  INCR_ANGLE  = 5
  TSDF_TRUNC  = 20

  WORLD_MIN = (0., 0.)
  WORLD_MAX = (PSIZE-1., PSIZE-1.)
  gp = ctimb.GridParams(WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1], PSIZE, PSIZE)

  tracker_params =  rt.RigidTrackerParams()
  tracker_params.tsdf_trunc_dist = TSDF_TRUNC

  def run(obs_num, img, init_phi, init_weight):
    state = ((img[:,:,0] != 255) | (img[:,:,1] != 255) | (img[:,:,2] != 255)).astype('float64')
    state = pad_state(state, NPAD)
    print state.shape
#     plt.subplot(211)
#     plt.imshow(state)
#     plt.subplot(212)
#     plt.imshow(pstate)
#     plt.show()

    tsdf, sdf, depth, w = observation.observation_from_full_state_rigid(state, tracker_params)
  
    ## optimize for camera pose and find the new sdf:
    new_phi, new_weight, obs_xy, problem_data = rt.run_one_rigid_step(gp, tracker_params, depth, tsdf, w, init_phi, init_weight, return_full=True)
    trusted = rt.threshold_trusted_for_view(new_weight)
    out_state = np.where(trusted, new_phi, np.nan)

    rt.plot_problem_data(plt, TSDF_TRUNC, gp, state, obs_xy, tsdf, w, init_phi, init_weight, new_phi, new_weight, out_state) 

    print problem_data['opt_result']['x']

    if args.output_dir is None and obs_num%10==0:
      plt.show()
    else:
      pass
      #plt.savefig('%s/plots_%d.png' % (args.output_dir, obs_num), bbox_inches='tight')

    if args.dump_dir is not None:
      import cPickle
      path = '%s/dump_%d.pkl' % (args.dump_dir, obs_num)
      with open(path, 'w') as f:
        cPickle.dump(problem_data, f, cPickle.HIGHEST_PROTOCOL)
      print 'wrote to', path

    return new_phi, new_weight

  orig_phi   = np.empty((PSIZE, PSIZE)); orig_phi.fill(tracker_params.tsdf_trunc_dist)
  orig_omega = np.zeros((PSIZE, PSIZE));

  def preprocess_img(img):
    return np.transpose(img, (1, 0, 2))

  if args.prior_img is not None:
    img = preprocess_img(ndimage.imread(args.prior_img))
    state = (img[:,:,0] == 0) & (img[:,:,1] == 0) & (img[:,:,2] == 0)
    sdf = ndimage.morphology.distance_transform_edt(~state)
    sdf[img[:,:,0] != 255] *= -1.
    orig_phi = np.clip(sdf, -tracker_params.tsdf_trunc_dist, tracker_params.tsdf_trunc_dist)
    orig_omega.fill(1)

  curr_phi, curr_omega = orig_phi, orig_omega

  if args.input_dir is not None:

    def sorted_nicely(l):
      """ Sort the given iterable in the way that humans expect.""" 
      import re
      convert = lambda text: int(text) if text.isdigit() else text 
      alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
      return sorted(l, key = alphanum_key)

    import os
    files = [(args.input_dir + '/' + f) for f in os.listdir(args.input_dir) if os.path.isfile(args.input_dir + '/' + f) and f.endswith('.png')]
    images = [preprocess_img(ndimage.imread(f)) for f in sorted_nicely(files)]
    
    for i, img in enumerate(images):
      curr_phi, curr_omega = run(i, img, curr_phi, curr_omega)

  else:
    orig_img = make_square_img(SIZE)
    for i in range(1500):
      angle = START_ANGLE + i*INCR_ANGLE
      img = ndimage.interpolation.rotate(orig_img, angle, cval=255, order=1, reshape=False)
      curr_phi, curr_omega = run(i, img, curr_phi, curr_omega)

if __name__ == '__main__':
  test_image()