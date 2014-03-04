import cPickle
from pprint import pprint
import timb
import rigid_tracker
import numpy as np
import os
import observation

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

  step_inds = np.floor(np.linspace(0, len(o['log'])-1, args.num_steps)).astype(int)
  entries = [cPickle.loads(fs.get(o['log'][i]['entry_file']).read()) for i in step_inds]

  import matplotlib
  matplotlib.rcParams.update({'font.size': 8, 'image.origin': 'lower'})
  import matplotlib.pyplot as plt

  def plot_binary(img):
    plt.imshow(img.T, aspect=1, origin='lower', cmap='binary', interpolation='bicubic')


  def plot_field(f, contour=False):
    plt.imshow(f.T, aspect=1, vmin=-10, vmax=10, cmap='bwr', origin='lower')
    # if contour:
    #   x = np.linspace(gp.xmin, gp.xmax, gp.nx)
    #   y = np.linspace(gp.ymin, gp.ymax, gp.ny)
    #   X, Y = np.meshgrid(x, y, indexing='ij')
    #   plt.contour(X, Y, f, levels=[0])

  ROWS = 4
  for i, e in enumerate(entries):
    plt.subplot(ROWS, len(entries), i+1 + len(entries)*0)
    plt.axis('off')
    plot_binary(e['state'])

    plt.subplot(ROWS, len(entries), i+1 + len(entries)*1)
    plt.axis('off')
    plot_field(e['new_phi'])

    plt.subplot(ROWS, len(entries), i+1 + len(entries)*2)
    plt.axis('off')
    plt.imshow(e['new_weight'].T, cmap='binary', vmin=0, vmax=observation.OBS_PEAK_WEIGHT*10).set_interpolation('nearest')

    plt.subplot(ROWS, len(entries), i+1 + len(entries)*3)
    plt.axis('off')
    trusted = timb.threshold_trusted_for_view2(e['new_weight'])
    machine_output = timb.sdf_to_zc(np.where(trusted, e['new_phi'], np.nan))
    plot_binary(machine_output)

  plt.show()

if __name__ == '__main__':
  main()
