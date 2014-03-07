import cPickle
from pprint import pprint
import timb
import rigid_tracker
import numpy as np
import os
import observation
import skimage.morphology

from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId

import argparse
parser = argparse.ArgumentParser(description='Plot sequence')
parser.add_argument('id')
parser.add_argument('--host', type=str)
parser.add_argument('--num_steps', type=int)
# parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

import skfmm
def get_sdf(img):
  """
  given an image (m,n,3) it returns the signed distance field.
  """
  
  def flood_fill(mask):
    """
    All entries inside the boundary are +1, outside -1.
    Starts filling from (0,0) [assumes (0,0) is outside.
    
     - Mask is a matrix of 0s and 1s.
     - Boundary is specified with 1s. Background with 0s.
    """
    def in_range(x,y):
      nx,ny = mask.shape
      return (0 <= x < nx) and (0 <= y < ny) 

    seg       = np.ones_like(mask, dtype='float64')
    processed = np.zeros_like(mask, dtype='bool')
    queue     = [(0,0)]

    while len(queue) != 0:
      cx,cy = queue.pop()
      processed[cx,cy] = True 

      ## check if in the background:
      if mask[cx,cy]==0: 
        seg[cx,cy] = -1
        ## add neighbors:
        for dx,dy in [(-1,0), (1,0), (0,-1), (0,1)]:
          mx,my = cx+dx, cy+dy
          if in_range(mx,my) and not processed[mx,my]:        
            queue.append([mx,my])
    return seg

  mask = ((img[:,:,0] != 255) | (img[:,:,1] != 255) | (img[:,:,2] != 255)).astype('float64')
  seg_mask = flood_fill(mask)
  sdf = skfmm.distance(seg_mask)
  return sdf

def main():
  # Connect to experiment db
  if args.host is not None:
    client = MongoClient(host=args.host)
  else:
    client = MongoClient()
  db = client.timb_experiments_db
  fs = gridfs.GridFS(db)

  # Pull out parts of the sequence
  o = db.experiments.find_one({'_id': ObjectId(args.id)})

  step_inds = np.floor(np.linspace(1, len(o['log'])-2, args.num_steps)).astype(int)
  entries = [cPickle.loads(fs.get(o['log'][i]['entry_file']).read()) for i in step_inds]

  import matplotlib
  matplotlib.rcParams.update({'font.size': 8, 'image.origin': 'lower'})
  import matplotlib.pyplot as plt

  # def plot_binary(img):
  #   img = skimage.morphology.skeletonize(img)
  #   plt.imshow(img.T, aspect=1, origin='lower', cmap='binary_r', interpolation='spline36')

  def plot_field(f, img=True, contour=False):
    assert img or contour
    if img:
      plt.imshow(f.T, aspect=1, vmin=-1, vmax=1, cmap='bwr', origin='lower')
    else:
      plt.imshow(np.zeros_like(f), aspect=1, vmin=-1, vmax=1, cmap='bwr', origin='lower')
    if contour:
      x = np.linspace(0, f.shape[0]-1, f.shape[0])
      y = np.linspace(0, f.shape[1]-1, f.shape[1])
      X, Y = np.meshgrid(x, y, indexing='ij')
      plt.contour(X, Y, f, levels=[0], colors='k')

  ROWS = 2
  plt.figure(figsize=(args.num_steps, ROWS), dpi=100)
  plt.subplots_adjust(wspace=.001, hspace=.001)
  for i, e in enumerate(entries):
    plt.subplot(ROWS, len(entries), i+1 + len(entries)*0)
    plt.axis('off')
    img = np.empty(e['state'].shape+(3,), dtype=np.uint8)
    for k in range(3):
      img[:,:,k] = np.where(e['state'], 0, 255)
    plot_field(get_sdf(img), False, True)

    # plt.subplot(ROWS, len(entries), i+1 + len(entries)*1)
    # plt.axis('off')
    # plot_field(e['new_phi'])

    # plt.subplot(ROWS, len(entries), i+1 + len(entries)*2)
    # plt.axis('off')
    # plt.imshow(e['new_weight'].T, cmap='binary_r', vmin=0, vmax=observation.OBS_PEAK_WEIGHT*10)

    plt.subplot(ROWS, len(entries), i+1 + len(entries)*1)
    plt.axis('off')
    trusted = timb.threshold_trusted_for_view2(e['new_weight'])
    machine_output = np.where(trusted, e['new_phi'], np.nan)
    # machine_output = timb.sdf_to_zc(np.where(trusted, e['new_phi'], np.nan))
    plot_field(machine_output, False, True)

  plt.savefig('out.pdf', bbox_inches='tight')
  plt.show()

if __name__ == '__main__':
  main()
