import numpy as np
import experiment
import timb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('experiment_class', type=str)
parser.add_argument('output_dir', default='.', type=str)
parser.add_argument('--show_plot', action='store_true')
args = parser.parse_args()

def main():
  mod_class_list = args.experiment_class.split('.')
  assert len(mod_class_list) == 2
  module_name, class_name = mod_class_list
  print 'Running experiment', class_name, 'in module', module_name
  import importlib
  mod = importlib.import_module(module_name)
  ex_class = getattr(mod, class_name)
  ex = ex_class()

  import matplotlib
  if not args.show_plot: matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  def iter_callback(i, data):
    timb.plot_problem_data(
      plt,
      tracker_params.tsdf_trunc_dist,
      ex.get_grid_params(),
      ex.get_state(i),
      data['obs_tsdf'],
      data['obs_weight'],
      data['curr_phi'],
      data['curr_weight'],
      data['problem_data']['result'],
      data['problem_data']['opt_result'],
      data['new_phi'],
      data['new_weight']
    )
    if args.output_dir is None:
      plt.show()
    else:
      plt.savefig('%s/plots_%d.png' % (args.output_dir, obs_num), bbox_inches='tight')


  ex_log = experiment.run_experiment(ex, tracker_params???, iter_callback)

  import os
  import cPickle
  import uuid
  output_filename = os.path.join(args.output_dir, uuid.uuid4() + '.log.pkl')
  print 'Writing to', output_filename
  with open(output_filename, 'w') as f:
    cPickle.dump(ex_log, f, cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
  main()
