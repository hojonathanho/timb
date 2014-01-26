import numpy as np
import experiment
import timb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output_dir', default='.', type=str)
parser.add_argument('experiment_class', type=str)
parser.add_argument('--argstr', type=str, required=True)
parser.add_argument('--show_plot', action='store_true')
parser.add_argument('--iter_cap', type=int, default=None)
args = parser.parse_args()

def run(tracker_params):
  mod_class_list = args.experiment_class.split('.')
  assert len(mod_class_list) == 2
  module_name, class_name = mod_class_list
  print 'Running experiment', class_name, 'in module', module_name
  import importlib
  mod = importlib.import_module(module_name)
  ex_class = getattr(mod, class_name)
  ex = ex_class.Create(args.argstr)

  # import matplotlib
  # if not args.show_plot: matplotlib.use('Agg')
  # import matplotlib.pyplot as plt

  def iter_callback(i, data):
    return
    # timb.plot_problem_data(
    #   plt,
    #   tracker_params.tsdf_trunc_dist,
    #   ex.get_grid_params(),
    #   ex.get_state(i),
    #   data['obs_tsdf'], data['obs_weight'],
    #   data['curr_phi'], data['curr_weight'],
    #   data['problem_data']['result'], data['problem_data']['opt_result'],
    #   data['new_phi'], data['new_weight'], data['output']
    # )
    # if args.output_dir is None:
    #   plt.show()
    # else:
    #   plt.savefig('%s/plots_%d.png' % (args.output_dir, i), bbox_inches='tight')

  ex_log = experiment.run_experiment(ex, tracker_params, iter_callback, args.iter_cap)

  import os
  import cPickle
  import uuid
  output_filename = os.path.join(args.output_dir, str(uuid.uuid4()) + '.log.pkl')
  print 'Writing to', output_filename
  out = {
    'tracker_params': tracker_params,
    'grid_params': ex.get_grid_params(),
    'log': ex_log,
  }
  with open(output_filename, 'w') as f:
    cPickle.dump(out, f, cPickle.HIGHEST_PROTOCOL)

def main():

  def gen_tracker_params():
    tracker_params = timb.TrackerParams()
    tracker_params.reweighting_iters = 10
    tracker_params.max_inner_iters = 20
    for a in [.1, 1.]:
      tracker_params.flow_rigidity_coeff = a
      for b in [True, False]:
        tracker_params.obs_weight_far = b
        for c in [True, False]:
          tracker_params.enable_smoothing = c
          for d in [True, False]:
            tracker_params.smoother_post_fmm = d
            yield tracker_params

  for p in gen_tracker_params():
    run(p)

if __name__ == '__main__':
  main()
