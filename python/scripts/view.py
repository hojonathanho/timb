import cPickle
from pprint import pprint
import timb
import numpy as np
import os


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_file', type=str)
parser.add_argument('--dump_to_dir', type=str, default=None)
parser.add_argument('--dump_to_mongo', action='store_true')
parser.add_argument('--mongo_experiment_name', type=str, default=None)
args = parser.parse_args()

import matplotlib
if args.dump_to_mongo or args.dump_to_dir is not None:
  matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


if args.dump_to_mongo:
  assert args.mongo_experiment_name is not None

  from pymongo import MongoClient
  client = MongoClient()
  db = client.timb_experiments_db
  import gridfs
  fs = gridfs.GridFS(db)

  gp_dict = {
    'xmin': grid_params.xmin,
    'xmax': grid_params.xmax,
    'ymin': grid_params.ymin,
    'ymax': grid_params.ymax,
    'nx': grid_params.nx,
    'ny': grid_params.ny,
    'eps_x': grid_params.eps_x,
    'eps_y': grid_params.eps_y,
  }

  doc = {
    'name': args.mongo_experiment_name,
    'tracker_params': tracker_params.__dict__,
    'grid_params': gp_dict,
    'log': []
  }


i = 0
while i < len(log):
  entry = log[i]

  trusted = timb.threshold_trusted_for_view(tracker_params, entry['new_phi'], entry['new_weight'])
  output = np.where(trusted, entry['new_phi'], np.nan)

  timb.plot_problem_data(
    plt,
    tracker_params.tsdf_trunc_dist,
    grid_params,
    entry['state'] if 'state' in entry else np.zeros_like(entry['obs_tsdf']),
    entry['obs_tsdf'], entry['obs_weight'],
    entry['curr_phi'], entry['curr_weight'],
    entry['problem_data']['result'], entry['problem_data']['opt_result'],
    entry['new_phi'], entry['new_weight'], output
  )

  if args.dump_to_dir is not None:
    matplotlib.use('Agg')
    print i
    plt.savefig(os.path.join(args.dump_to_dir, '%05d.png' % i), bbox_inches='tight')
    i += 1

  elif args.dump_to_mongo:
    print i
    import uuid
    name = os.path.join('/tmp', str(uuid.uuid4()) + '.png')
    plt.savefig(name, bbox_inches='tight')
    with open(name, 'rb') as f:
      plot_file = fs.put(f.read())

    entry_file = fs.put(cPickle.dumps(entry, protocol=2))

    doc['log'].append({
      'entry_file': entry_file,
      'plot_file': plot_file
    })

    i += 1

  else:
    plt.show(block=False)
    x = raw_input('> ')
    if x == 'b':
      i = (i - 1) % len(log)
    else:
      i += 1
      if i == len(log):
        break

if args.dump_to_mongo:
  print db.experiments.insert(doc)
