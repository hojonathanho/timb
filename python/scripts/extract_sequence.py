import cPickle
from pprint import pprint
import timb
import rigid_tracker
import numpy as np
import os
import scipy.io
import subprocess
import uuid

from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('id')
parser.add_argument('--host', type=str)
# parser.add_argument('--i', type=int, required=True)
# parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()


def sdf_to_zc(f):
  p = np.pad(f, (1,1), 'edge')
  return (f*p[:-2,1:-1] < 0) | (f*p[2:,1:-1] < 0) | (f*p[1:-1,:-2] < 0) | (f*p[1:-1,2:] < 0)


def main():
  # Connect to experiment db
  if args.host is not None:
    client = MongoClient(host=args.host)
  else:
    client = MongoClient()
  db = client.timb_experiments_db
  fs = gridfs.GridFS(db)

  # Pull out input/output sequence
  out_bmaps = []
  out_true_bmaps = []

  o = db.experiments.find_one({'_id': ObjectId(args.id)})
  for i, log_entry in enumerate(o['log']):
    entry = cPickle.loads(fs.get(log_entry['entry_file']).read())

    out_true_bmaps.append(entry['state'])

    trusted = timb.threshold_trusted_for_view2(entry['new_weight'])
    machine_output = sdf_to_zc(np.where(trusted, entry['new_phi'], np.nan))
    out_bmaps.append(machine_output)

  out_bmaps = np.array(out_bmaps)
  out_true_bmaps = np.array(out_true_bmaps)

  # Write i/o sequence to matlab
  input_filename = '/tmp/%s.mat' % str(uuid.uuid4())
  output_filename = '/tmp/%s.mat' % str(uuid.uuid4())
  scipy.io.savemat(input_filename, {'bmaps': out_bmaps, 'true_bmaps': true_bmaps})

  # Call matlab function to calculuate PR
  subprocess.call(['matlab', '-nojvm', '-nodisplay', '-nosplash', '-nodesktop', '-r', 
    'multiPR(%s, %s); exit' % (input_filename, output_filename)
  ])

  output = scipy.io.loadmat(output_filename)
  os.unlink(input_filename)
  os.unlink(output_filename)


if __name__ == '__main__':
  main()
