import cPickle
from pprint import pprint
import timb
import numpy as np

import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_file', type=str)
args = parser.parse_args()

with open(args.data_file, 'rb') as f:
  data_str = f.read()
data = cPickle.loads(data_str)
del data_str

print 'Tracker params:'
tracker_params = data['tracker_params']
pprint(tracker_params.__dict__)

log = data['log']
print 'Log has %d steps' % len(log)

if 'grid_params' in data:
  grid_params = data['grid_params']
else:
  grid_params = timb.GridParams(0, 99, 0, 99, 100, 100)

for i, entry in enumerate(log):
  print i

  trusted = timb.threshold_trusted_for_view(tracker_params, entry['new_phi'], entry['new_weight'])
  output = np.where(trusted, entry['new_phi'], np.nan)

  timb.plot_problem_data(
    plt,
    tracker_params.tsdf_trunc_dist,
    grid_params,
    np.zeros_like(entry['obs_tsdf']),
    entry['obs_tsdf'], entry['obs_weight'],
    entry['curr_phi'], entry['curr_weight'],
    entry['problem_data']['result'], entry['problem_data']['opt_result'],
    entry['new_phi'], entry['new_weight'], output
  )
  plt.show(block=False)

  x = raw_input('> ')

#import IPython; IPython.embed()
