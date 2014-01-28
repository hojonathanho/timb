import numpy as np
import experiment
import timb
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output_dir', default='.', type=str)
parser.add_argument('experiment_class', type=str)
parser.add_argument('--argstr', type=str, required=True)
parser.add_argument('--show_plot', action='store_true')
parser.add_argument('--iter_cap', type=int, default=None)
parser.add_argument('--parallel', type=int, default=None)
parser.add_argument('--fake', action='store_true')
args = parser.parse_args()

def run(tracker_params):
  if args.fake:
    print tracker_params.flow_rigidity_coeff, tracker_params.enable_smoothing, tracker_params.smoother_phi_ignore_thresh, tracker_params.smoother_weight_ignore_thresh
    return

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

  t_start = time.time()
  ex_log = experiment.run_experiment(ex, tracker_params, iter_callback, args.iter_cap)
  t_elapsed = time.time() - t_start

  import os
  import cPickle
  import uuid
  from datetime import datetime
  output_filename = os.path.join(args.output_dir, str(uuid.uuid4()) + '.log.pkl')
  print 'Writing to', output_filename
  out = {
    'tracker_params': tracker_params,
    'grid_params': ex.get_grid_params(),
    'log': ex_log,
    'datetime': datetime.now(),
    'time_elapsed': t_elapsed,
  }
  with open(output_filename, 'w') as f:
    cPickle.dump(out, f, cPickle.HIGHEST_PROTOCOL)

def main():

  def gen_params():
    from copy import deepcopy
    tracker_params = timb.TrackerParams()
    tracker_params.reweighting_iters = 10
    tracker_params.max_inner_iters = 10
    tracker_params.obs_weight_far = True
    tracker_params.smoother_post_fmm = False

    for a in [.1, 1.]:
      tracker_params.flow_rigidity_coeff = a
      for b in [True]:
        tracker_params.enable_smoothing = b
        if not tracker_params.enable_smoothing:
          yield deepcopy(tracker_params)
        else:
          for c in [5., 8.]:
            tracker_params.smoother_phi_ignore_thresh = c
            for d in [1e-2, 1e-1]:
              tracker_params.smoother_weight_ignore_thresh = d
              yield deepcopy(tracker_params)


# def gen_params():
#   from copy import deepcopy
#   tracker_params = timb.TrackerParams()
#   tracker_params.reweighting_iters = 10
#   tracker_params.max_inner_iters = 10
#   for a in [.1, 1.]:
#     tracker_params.flow_rigidity_coeff = a
#     for b in [True, False]:
#       tracker_params.obs_weight_far = b
#       for c in [True, False]:
#         tracker_params.enable_smoothing = c
#         for d in [True, False]:
#           tracker_params.smoother_post_fmm = d
#           yield deepcopy(tracker_params)


  if args.parallel is None:
    for p in gen_params():
      run(p)
  else:
    from joblib import Parallel, delayed
    Parallel(n_jobs=args.parallel)(delayed(run)(params) for params in gen_params())

if __name__ == '__main__':
  main()
