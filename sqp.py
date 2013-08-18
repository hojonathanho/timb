import numpy as np
import optimization
import time

GRID_NX = 50
GRID_NY = 50
GRID_SHAPE = (GRID_NX, GRID_NY)

def make_varname_mat(prefix, shape):
  import itertools
  out = np.empty(shape, dtype=object)
  for inds in itertools.product(*[range(d) for d in shape]):
    out[inds] = prefix + '_' + '_'.join(map(str, inds))
  return out

def sumsq(a):
  return (a**2).sum()

def cost(phi, u):
  out = sumsq(u_x) + sumsq(u_y)
  out += np.gradient(u_x)

def construct_prob():
  opt = optimization.GurobiSQP(None)

  t_start = time.time() 
  phi_names = make_varname_mat('phi', GRID_SHAPE)
  u_names = make_varname_mat('u', GRID_SHAPE + (2,))
  phi_vars = opt.add_vars(phi_names)
  u_vars = opt.add_vars(u_names)

  opt.obj_convex_fn = lambda _: sumsq(phi_vars) + sumsq(u_vars)

  print 'built problem in', time.time()-t_start
  opt.optimize(None)
  print opt.get_values(phi_vars)
  print opt.get_values(u_vars)

if __name__ == '__main__':
  construct_prob()
