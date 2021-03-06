import numpy as np
import scipy.optimize as sio
import gurobipy
import collections
import logging

def gradient_descent(fn, fn_grad, x0, gtol=1e-5, maxiter=100):
  i = 0
  x = x0.copy()
  while i < maxiter:
    i += 1
    dx = -fn_grad(x)
    if abs(dx).max() <= gtol:
      print 'Terminated since |g| <= %f' % gtol
      break
    t = sio.line_search(fn, fn_grad, x, dx, -dx)[0]
    x += t*dx
    print 'Step %d: y=%f, |g|=%f, t=%f' % (i, fn(x), np.linalg.norm(dx), t)
  if i >= maxiter:
    print 'Terminated due to iteration limit'
  return x

import time
class Timer(object):
  def __init__(self):
    self.times = collections.defaultdict(float)
    self.curr = {}

  def start(self, name):
    assert name not in self.curr
    self.curr[name] = time.time()

  def end(self, name):
    assert name in self.curr
    self.times[name] += time.time() - self.curr[name]
    del self.curr[name]

  def display(self):
    print 'Timing information:'
    total = sum(self.times.values())
    for name, t in self.times.iteritems():
      print '\t%s:\t%f (%f%%)' % (name, t, 100.*t/total)
    print '\tTotal:\t%f' % total

class CostFunc(object):
  def get_name(self): raise NotImplementedError
  def get_vars(self): raise NotImplementedError
  def eval(self, vals): raise NotImplementedError
  def convex(self, vals): raise NotImplementedError

GurobiSQPResult = collections.namedtuple('GurobiSQPResult', [
  'status',
  'info',
  'x',
  'cost',
  'cost_detail',
  'x_over_iters',
  'costs_over_iters',
])
class GurobiSQP(object):
  def __init__(self):
    # Algorithm parameters
    self.init_trust_region_size = 1.
    self.trust_shrink_ratio, self.trust_expand_ratio = .1, 2.
    self.min_trust_region_size = 1e-4
    self.min_approx_improve = 1e-6
    self.improve_ratio_threshold = .25
    self.max_iter = 50

    self.all_vars, self.varname2ind, self.input_var_ordering = [], {}, None
    self.costs = []
    self.model = gurobipy.Model('qp')
    self.model.setParam('OutputFlag', False)

    self.callbacks = []

    self.logger = logging.getLogger('sqp')
    self.logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter('>>> %(name)s - %(levelname)s - %(message)s'))
    self.logger.addHandler(ch)

    self.timer = Timer()

  def add_vars(self, names):
    def _add_var(name):
      assert name not in self.varname2ind
      v = self.model.addVar(lb=-gurobipy.GRB.INFINITY, ub=gurobipy.GRB.INFINITY, name=name)
      self.varname2ind[name] = len(self.all_vars)
      self.all_vars.append(v)
      return v
    vars = np.frompyfunc(_add_var, 1, 1)(names)
    self.model.update()
    return vars

  def add_cost(self, cost):
    self.costs.append(cost)

  @staticmethod
  def _get_curr_var_values(vars):
    return np.frompyfunc(lambda v: v.x, 1, 1)(vars).astype(float)

  def _get_var_values_at(self, vars, x0):
    assert len(x0) == len(self.all_vars)
    return np.frompyfunc(lambda v: x0[self.varname2ind[v.varName]], 1, 1)(vars).astype(float)

  def _build_convex_objective(self, x0):
    assert len(x0) == len(self.all_vars)
    obj = 0
    detail = {}
    for c in self.costs:
      part = c.convex(self._get_var_values_at(c.get_vars(), x0))
      detail[c.get_name()] = part
      obj += part
    return obj, detail

  def _eval_true_objective(self, x0):
    assert len(x0) == len(self.all_vars)
    val = 0
    detail = {}
    for c in self.costs:
      v = c.eval(self._get_var_values_at(c.get_vars(), x0))
      detail[c.get_name()] = v
      val += v
    return val, detail

  def _eval_curr_model_objective(self, convex_obj_detail):
    val = 0
    detail = {}
    for c in self.costs:
      self.logger.debug(c.get_name())
      v = convex_obj_detail[c.get_name()].getValue()
      detail[c.get_name()] = v
      val += v
    return val, detail

  def _set_trust_region(self, size, x0):
    assert len(x0) == len(self.all_vars)
    for v, x in zip(self.all_vars, x0):
      v.lb, v.ub = float(x) - size, float(x) + size

  def add_callback(self, cb):
    self.callbacks.append(cb)

  def declare_input_var_ordering(self, vars):
    self.input_var_ordering = np.array([self.varname2ind[v.varName] for v in vars], dtype=int)
    print self.input_var_ordering
    assert len(set(self.input_var_ordering)) == len(self.all_vars)

  def optimize(self, start_x):
    assert self.input_var_ordering is not None
    assert (self.input_var_ordering == np.sort(self.input_var_ordering)).all() # TODO: implement reordering
    #start_x = np.asarray(start_x)[self.input_var_ordering]

    info = {
      'n_qp_solves': 0,
      'n_func_evals': 0,
      'n_iters': 0
    }
    status = 'incomplete'

    exit = False
    curr_x = start_x
    curr_cost, curr_cost_detail = self._eval_true_objective(start_x)
    curr_iter = 0
    trust_region_size = self.init_trust_region_size
    costs_over_iters, x_over_iters = [], []
    while True:

      curr_result = GurobiSQPResult(status=status, info=info, x=curr_x, cost=curr_cost, cost_detail=curr_cost_detail, x_over_iters=x_over_iters, costs_over_iters=costs_over_iters)
      for cb in self.callbacks:
        cb(curr_result)

      curr_iter += 1
      costs_over_iters.append(curr_cost)
      x_over_iters.append(curr_x)
      self.logger.info('Starting SQP iteration %d' % curr_iter)

      self.logger.debug('Convexifying objective')
      self.timer.start('convexify')
      curr_convex_obj, curr_convex_obj_detail = self._build_convex_objective(curr_x)
      self.logger.debug('Setting Gurobi objective')
      self.model.setObjective(curr_convex_obj)
      self.timer.end('convexify')

      while trust_region_size >= self.min_trust_region_size:
        self._set_trust_region(trust_region_size, curr_x)
        self.logger.debug('Solving QP')
        self.timer.start('solve_qp')
        self.model.optimize()
        self.timer.end('solve_qp')
        info['n_qp_solves'] += 1
        # TODO: ERROR CHECK QP

        self.timer.start('extract')
        self.logger.debug('Extracting model values')
        new_x = self._get_curr_var_values(self.all_vars)
        self.logger.debug('Extracting model costs')
        model_cost, model_cost_detail = self._eval_curr_model_objective(curr_convex_obj_detail)
        self.logger.debug('Evaluating true objective')
        new_cost, new_cost_detail = self._eval_true_objective(new_x)
        info['n_func_evals'] += 1
        self.logger.debug('Done')
        self.timer.end('extract')

        approx_merit_improve = curr_cost - model_cost
        exact_merit_improve = curr_cost - new_cost

        self.logger.info("%15s | %10s | %10s | %10s | %10s" % ("name", "oldexact", "dapprox", "dexact", "ratio"))
        for cost in self.costs:
          approx_improve = curr_cost_detail[cost.get_name()] - model_cost_detail[cost.get_name()]
          exact_improve = curr_cost_detail[cost.get_name()] - new_cost_detail[cost.get_name()]
          ratio = np.nan if np.allclose(approx_improve, 0) else exact_improve/approx_improve
          self.logger.info('%15s | %10.3e | %10.3e | %10.3e | %10.3e' % (cost.get_name(), curr_cost_detail[cost.get_name()], approx_improve, exact_improve, ratio))

        if approx_merit_improve < -1e-5:
          self.logger.warn("approximate merit function got worse (%.3e). (convexification is probably wrong to zeroth order)" % approx_merit_improve)

        if approx_merit_improve < self.min_approx_improve:
          self.logger.info("converged because improvement was small (%.3e < %.3e)" % (approx_merit_improve, self.min_approx_improve))
          status = 'converged'
          exit = True
          break

        merit_improve_ratio = exact_merit_improve / approx_merit_improve
        if exact_merit_improve < 0 or merit_improve_ratio < self.improve_ratio_threshold:
          trust_region_size *= self.trust_shrink_ratio
          self.logger.info("shrunk trust region. new box size: %.4f" % trust_region_size)
        else:
          curr_x, curr_cost, curr_cost_detail = new_x, new_cost, new_cost_detail
          trust_region_size *= self.trust_expand_ratio
          self.logger.info("expanded trust region. new box size: %.4f" % trust_region_size)
          break

      if exit:
        break

      if trust_region_size < self.min_trust_region_size:
        self.logger.info("converged because trust region is tiny")
        status = 'converged'
        exit = True
        break

      if curr_iter >= self.max_iter:
        self.logger.warn("iteration limit")
        status = 'iter_limit'
        exit = True
        break

    assert exit
    info['n_iters'] = curr_iter
    costs_over_iters.append(curr_cost)
    x_over_iters.append(curr_x)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # plt.plot(np.arange(len(costs_over_iters)), np.log(costs_over_iters))
    # plt.show()
    self.timer.display()
    return GurobiSQPResult(status=status, info=info, x=curr_x, cost=curr_cost, cost_detail=curr_cost_detail, x_over_iters=x_over_iters, costs_over_iters=costs_over_iters)
