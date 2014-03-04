import cPickle
from pprint import pprint
import timb
import rigid_tracker
import numpy as np
import os

from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('id')
parser.add_argument('--host', type=str)
parser.add_argument('--i', type=int, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

if args.host is not None:
  client = MongoClient(host=args.host)
else:
  client = MongoClient()
db = client.timb_experiments_db
fs = gridfs.GridFS(db)

o = db.experiments.find_one({'_id': ObjectId(args.id)})

entry = cPickle.loads(fs.get(o['log'][args.i]['entry_file']).read())

state = entry['state']

def sdf_to_zc(f):
  p = np.pad(f, (1,1), 'edge')
  return (f*p[:-2,1:-1] < 0) | (f*p[2:,1:-1] < 0) | (f*p[1:-1,:-2] < 0) | (f*p[1:-1,2:] < 0)

trusted = timb.threshold_trusted_for_view2(entry['new_weight'])
output = sdf_to_zc(np.where(trusted, entry['new_phi'], np.nan))

import scipy.io
scipy.io.savemat(args.output, {'machine':output, 'true':state})


print 'state', state, state.shape
print 'output', output, output.shape


import sys
sys.exit(0)

import matplotlib
matplotlib.rcParams.update({'font.size': 8, 'image.origin': 'lower'})
import matplotlib.pyplot as plt


# def saveFigureAsImage(fileName,fig=None,**kwargs):
#     ''' Save a Matplotlib figure as an image without borders or frames.
#        Args:
#             fileName (str): String that ends in .png etc.

#             fig (Matplotlib figure instance): figure you want to save as the image
#         Keyword Args:
#             orig_size (tuple): width, height of the original image used to maintain 
#             aspect ratio.
#     '''
#     fig_size = fig.get_size_inches()
#     w,h = fig_size[0], fig_size[1]
#     fig.patch.set_alpha(0)
#     if kwargs.has_key('orig_size'): # Aspect ratio scaling if required
#         w,h = kwargs['orig_size']
#         w2,h2 = fig_size[0],fig_size[1]
#         fig.set_size_inches([(w2/w)*w,(w2/w)*h])
#         fig.set_dpi((w2/w)*fig.get_dpi())
#     a=fig.gca()
#     a.set_frame_on(False)
#     a.set_xticks([]); a.set_yticks([])
#     plt.axis('off')
#     plt.xlim(0,h); plt.ylim(w,0)
#     fig.savefig(fileName, transparent=True, bbox_inches='tight', \
#                         pad_inches=0)

def plot_field(f, contour=False):
  # plt.imshow(f.T, aspect=1, vmin=-10, vmax=10, cmap='bwr', origin='lower')
  # if contour:
  x = np.linspace(0, f.shape[0]-1, f.shape[0])
  y = np.linspace(0, f.shape[1]-1, f.shape[1])
  X, Y = np.meshgrid(x, y, indexing='ij')
  plt.contour(X, Y, f, levels=[0], colors='k')

print state.astype(float)
# plt.figure(figsize=(1,1), dpi=100)
# plt.axis('off')
# Z = np.empty((100,100,3))
# Z[:,:,0] = Z[:,:,1] = Z[:,:,2] = (~state).T.astype(float)
# plt.imshow(Z, aspect=1, origin='lower', interpolation='none')

fig = plt.figure(figsize=(1,1), dpi=100)
plt.axis('off')
plot_field(output, True)
plt.show()

fig = plt.figure(figsize=(1,1), dpi=100)
plt.axis('off')
plt.imshow(sdf_to_zc(output).T, aspect=1, origin='lower')
plt.show()

