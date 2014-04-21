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
parser = argparse.ArgumentParser(description='Extract input/output sequence and plot F1 score over time')
parser.add_argument('id')
parser.add_argument('--host', type=str)
# parser.add_argument('--i', type=int, required=True)
# parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

MATLAB_CWD = '/home/jonathan/Downloads/segbench/Benchmark'

def main():
  # Connect to experiment db
  if args.host is not None:
    client = MongoClient(host=args.host)
  else:
    client = MongoClient()
  db = client.timb_experiments_db
  fs = gridfs.GridFS(db)

  # Pull out input/output sequence
  o = db.experiments.find_one({'_id': ObjectId(args.id)})
  from pprint import pprint
  pprint(o['tracker_params'])

  out_bmaps = np.empty((100, 100, len(o['log'])), dtype=np.uint32)
  out_true_bmaps = np.empty((100, 100, len(o['log'])), dtype=np.uint32)

  for i, log_entry in enumerate(o['log']):
    entry = cPickle.loads(fs.get(log_entry['entry_file']).read())
    out_true_bmaps[:,:,i] = entry['state']

    trusted = timb.threshold_trusted_for_view2(entry['new_weight'])
    machine_output = timb.sdf_to_zc(np.where(trusted, entry['new_phi'], np.nan))
    out_bmaps[:,:,i] = machine_output

  # out_bmaps = np.array(out_bmaps)
  # out_true_bmaps = np.array(out_true_bmaps)

  # Write i/o sequence to matlab
  input_filename = '/tmp/%s.mat' % str(uuid.uuid4())
  output_filename = '/tmp/%s.mat' % str(uuid.uuid4())
  scipy.io.savemat(input_filename, {'bmaps': out_bmaps, 'true_bmaps': out_true_bmaps})

  # Call matlab function to calculuate PR
  subprocess.call(['matlab', '-nosplash', '-nodesktop', #'-nojvm', '-nodisplay', 
    '-r', "multiPR('%s', '%s'); exit" % (input_filename, output_filename)
  ], cwd=MATLAB_CWD)
  output = scipy.io.loadmat(output_filename)
  os.unlink(input_filename)
  os.unlink(output_filename)

  out_ps = np.squeeze(output['ps'])
  out_rs = np.squeeze(output['rs'])
  out_f1s = np.squeeze(output['f1s'])

  print out_f1s

  print 'average f1', np.mean(out_f1s[np.isfinite(out_f1s)])

  import matplotlib.pyplot as plt
  plt.plot(out_ps, label="precision")
  plt.plot(out_rs, label="recall")
  plt.plot(out_f1s, label="f1")
  plt.legend(loc=4)
  plt.ylim([0, 1])
  plt.show()

if __name__ == '__main__':
  main()
