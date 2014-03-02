import numpy as np
import os
from scipy import ndimage
from scipy.misc import imsave

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('out_dir')
parser.add_argument('--start_angle', default=0, type=int)
parser.add_argument('--incr_angle', default=5, type=int)
parser.add_argument('--num_steps', default=150, type=int)
args = parser.parse_args()

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

def main():
  out_dir = os.path.expanduser(args.out_dir)
  assert os.path.isdir(out_dir)
  assert len(os.listdir(out_dir)) == 0

  SIZE = 100
  orig_img = make_square_img(SIZE)

  for i in range(args.num_steps):
    angle = args.start_angle + i*args.incr_angle
    img = ndimage.interpolation.rotate(orig_img, angle, cval=255, order=1, reshape=False)
    filename = os.path.join(out_dir, '%05d.png' % i)
    imsave(filename, img)
    print 'Wrote', filename

if __name__ == '__main__':
  main()
