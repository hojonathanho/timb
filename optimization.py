import numpy as np
import scipy.optimize as sio
import gurobipy
from gurobipy import GRB
import sympy
import collections

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



class GurobiSQP(object):
  def __init__(self, symbols, obj_convex_fn):
    '''
    obj_convex_fn: function that takes a point and returns the objective (symbolic expr) convexified around that point
    '''
    self.obj_convex_fn = obj_convex_fn

    self.model = gurobipy.Model('qp')
    self.sym2var = collections.OrderedDict()
    assert len(set(s.name for s in symbols)) == len(symbols)
    for s in symbols:
      self.sym2var[s] = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=s.name)
    self.model.update()

  def _sympy_to_gurobi(self, expr):
    if isinstance(expr, sympy.Symbol):
      return self.sym2var[expr]
    elif isinstance(expr, sympy.Number):
      return float(expr)
    elif isinstance(expr, sympy.Add):
      return sum(self._sympy_to_gurobi(a) for a in expr.args)
    elif isinstance(expr, sympy.Mul):
      acc = 1
      for a in expr.args:
        acc *= self._sympy_to_gurobi(a)
      return acc
    elif isinstance(expr, sympy.Pow):
      assert expr.args[1] == 2
      base = self._sympy_to_gurobi(expr.args[0])
      return base * base
    else:
      raise NotImplementedError('Don\'t know how to convert %s to Gurobi (expr: %s)' % (expr.func, expr))

  def get_value(self):
    vals = np.empty(len(self.sym2var))
    for i, var in enumerate(self.sym2var.itervalues()):
      vals[i] = var.x
    return vals

  def optimize(self, start_val):
    val = start_val

    sym_convex_obj = self.obj_convex_fn(val)#.expand()
    convex_obj = self._sympy_to_gurobi(sym_convex_obj)
    #print 'converted sympy', sym_convex_obj, 'to', convex_obj

    self.model.setObjective(convex_obj)
    self.model.optimize()
    return self.get_value()

def sym_grad(expr, wrt):
  g = np.empty(len(wrt), dtype=np.object)
  for i, v in enumerate(wrt):
    print i, len(wrt)
    g[i] = expr.diff(v)
  return g

if __name__ == '__main__':
  shape = (40, 40)
  syms = sympy.symarray('m', shape)
  centers = np.random.rand(*shape)
  print 'Building objective expr'
  obj = ((syms - centers)**2).sum()
  print 'done'
  import IPython; IPython.embed()
  opt = GurobiSQP(syms.ravel(), lambda _: obj)
  result = opt.optimize(start_val=np.zeros(shape).ravel())
  print result
  print 'ok?', np.allclose(result.reshape(shape), centers)


# s_x = sympy.Symbol('x')
# s_y = sympy.Symbol('y')

# obj_fn = lambda _: ((s_x - 1)**2 + (s_y + 3)**2)
# opt = GurobiSQP([s_x, s_y], obj_fn)
# print opt.optimize(start_val=[0,0])
