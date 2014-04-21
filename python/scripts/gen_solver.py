import numpy as np
import random
import sympy
import scipy.weave

def random_str(length):
  import string
  return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))


def finite_diff(xm1, x0, x1, h, mode):
  '''x0: left, x1: middle, x2: right, h: spacing
  mode: c: central, f: forward, b: backward
  '''
  if mode == 'c':
    return (x1 - xm1) / (2*h)
  if mode == 'f':
    return (x1 - x0) / h
  if mode == 'b':
    return (x0 - xm1) / h
  raise NotImplementedError


def finite_diff_2(xm2, xm1, x0, x1, x2, h, mode):
  '''
  second derivative
  x0: left, x1: middle, x2: right, h: spacing
  mode: c: central, f: forward, b: backward
  '''
  if mode == 'c':
    return (xm1 - 2*x0 + x1) / (h*h)
  if mode == 'f':
    return (x0 - 2*x1 + x2) / (h*h)
  if mode == 'b':
    return (x0 - 2*xm1 + xm2) / (h*h)
  raise NotImplementedError


def make_var_symbols(name, order):
  assert order >= 1

  def n_to_offset(prefix, n):
    if n == 0: return prefix
    if n > 0: return '%s+%d' % (prefix, n)
    return '%s-%d' % (prefix, -n)

  size = 2*order + 1

  a = np.empty((size,size), dtype=object)
  for i in range(size):
    for j in range(size):
      a[i,j] = sympy.Symbol('%s(%s,%s)' % (name, n_to_offset('i', i-order), n_to_offset('j', j-order)))
  return a


class FieldVars(object):
  def __init__(self, name, hx, hy, order=2, latex_name=None):
    self.name = name
    self.order = order
    self.internal_name = random_str(10)
    self.latex_name = name if latex_name is None else latex_name # todo
    self.u = make_var_symbols(self.internal_name, order)
    self.hx, self.hy = hx, hy

    self.mode_x, self.mode_y = None, None

  def __getitem__(self, idx):
    assert len(idx) == 2
    return self.u[idx[0]+self.order, idx[1]+self.order]

  def set_x_mode(self, mode_x):
    self.mode_x = mode_x

  def set_y_mode(self, mode_y):
    self.mode_y = mode_y

  def dx(self):
    return finite_diff(self[-1,0], self[0,0], self[1,0], self.hx, self.mode_x)

  def dy(self):
    return finite_diff(self[0,-1], self[0,0], self[0,1], self.hy, self.mode_y)

  def dx2(self):
    return finite_diff_2(self[-2,0], self[-1,0], self[0,0], self[1,0], self[2,0], self.hx, self.mode_x)

  def dy2(self):
    return finite_diff_2(self[0,-2], self[0,-1], self[0,0], self[0,1], self[0,2], self.hy, self.mode_y)

  def dxdy(self):
    return finite_diff(
      finite_diff(self[-1,-1], self[-1,0], self[-1,1], self.hy, self.mode_y),
      finite_diff(self[0,-1], self[0,0], self[0,1], self.hy, self.mode_y),
      finite_diff(self[1,-1], self[1,0], self[1,1], self.hy, self.mode_y),
      self.hx, self.mode_x
    )

  # def replace_var_names(self, s, style='c'):
  #   if style == 'c':
  #     return s.replace(self.internal_name, self.name)
  #   elif style == 'latex':
  #     return s.replace(self.internal_name, self.latex_name)
  #   else:
  #     raise NotImplementedError


# def solve_by_factorization(eval_expr_func, var_grids, grid_size):
#   assert len(var_grid) == 1 # heh
#   # Make variable symbols for every point in the grid
#   syms = {}
#   for vg in var_grids:
#     s = np.zeros((grid_size, grid_size), dtype=object)
#     for i in range(grid_size):
#       for j in range(grid_size):
#         s[i,j] = sympy.Symbol('%s_{%d,%d}' % (vg.name, i, j))
#     syms[vg] = s

#   # Form the equations for each grid point
#   equations = {}
#   for vg in var_grids:
#     eqns = np.zeros((grid_size, grid_size), dtype=object)
#     for i in range(grid_size):
#       for j in range(grid_size):
#         # Set finite difference mode in x direction
#         if i == 0:
#           for vg in var_grids: vg.set_x_mode('f')
#         elif i == grid_size-1:
#           for vg in var_grids: vg.set_x_mode('b')
#         else:
#           vg.set_x_mode('c')

#         # Set finite difference mode in y direction
#         if j == 0:
#           for vg in var_grids: vg.set_y_mode('f')
#         elif j == grid_size-1:
#           for vg in var_grids: vg.set_y_mode('b')
#         else:
#           vg.set_y_mode('c')

#         e = eval_expr_func()
#         eqns[i,j] = e

#     equations[vg] = eqns

def make_grid_iter_code(expr_dict, var_name_1, var_ptr_name_1, var_name_2, var_ptr_name_2, num_iters_name, grid_size_name):
  return r'''
  blitz::Array<double, 2> *{var_ptr_name_1} = &{var_name_1};
  blitz::Array<double, 2> *{var_ptr_name_2} = &{var_name_2};

  for (int iter = 0; iter < {num_iters}; ++iter) {{

    if (iter != 0) {{
      std::swap({var_ptr_name_1}, {var_ptr_name_2});
    }}

    for (int i = 0; i < {grid_size}; ++i) {{
      for (int j = 0; j < {grid_size}; ++j) {{
        if (i == 0) {{

          if (j == 0) {{
            (*{var_ptr_name_2})(i,j) = {exprs[ff]};
          }} else if (j == grid_size-1) {{
            (*{var_ptr_name_2})(i,j) = {exprs[fb]};
          }} else {{
            (*{var_ptr_name_2})(i,j) = {exprs[fc]};
          }}

        }} else if (i == grid_size-1) {{

          if (j == 0) {{
            (*{var_ptr_name_2})(i,j) = {exprs[bf]};
          }} else if (j == grid_size-1) {{
            (*{var_ptr_name_2})(i,j) = {exprs[bb]};
          }} else {{
            (*{var_ptr_name_2})(i,j) = {exprs[bc]};
          }}

        }} else {{

          if (j == 0) {{
            (*{var_ptr_name_2})(i,j) = {exprs[cf]};
          }} else if (j == grid_size-1) {{
            (*{var_ptr_name_2})(i,j) = {exprs[cb]};
          }} else {{
            (*{var_ptr_name_2})(i,j) = {exprs[cc]};
          }}

        }}
      }}
    }}

  }}

  return_val = {var_ptr_name_2} == &{var_name_1};
  '''.format(
    exprs=expr_dict,
    num_iters=num_iters_name, grid_size=grid_size_name,
    var_name_1=var_name_1, var_ptr_name_1=var_ptr_name_1,
    var_name_2=var_name_2, var_ptr_name_2=var_ptr_name_2
  )


class Jacobi(object):
  def __init__(self, create_expr_func, field):
    self.create_expr_func, self.field = create_expr_func, field

    # compute expr for all settings of boundaries
    self.var_name_1 = 'u_1'
    self.var_ptr_name_1 = 'p_u_1'
    self.var_name_2 = 'u_2'
    self.var_ptr_name_2 = 'p_u_2'
    self.exprs = {}
    for x in 'bcf':
      for y in 'bcf':
        field.set_x_mode(x)
        field.set_y_mode(y)
        ccode = sympy.printing.ccode(self.create_expr_func())
        self.exprs[x+y] = ccode.replace(field.internal_name, '(*%s)' % self.var_ptr_name_1)

    for k, v in self.exprs.iteritems(): print k, v

    self.jacobi_code = make_grid_iter_code(self.exprs, self.var_name_1, self.var_ptr_name_1, self.var_name_2, self.var_ptr_name_2, 'num_iters', 'grid_size')
    print self.jacobi_code


  def run(self, u_input, num_iters):
    assert u_input.shape[0] == u_input.shape[1]
    grid_size = u_input.shape[0]

    u1 = u_input.astype(float).copy()
    u2 = u1.copy()

    local_dict = {self.var_name_1: u1, self.var_name_2: u2, 'grid_size': grid_size, 'num_iters': num_iters}
    last_is_u1 = scipy.weave.inline(
      self.jacobi_code,
      local_dict.keys(),
      local_dict,
      type_converters=scipy.weave.converters.blitz
    )
    return u1 if last_is_u1 else u2


def jacobi_laplace_equation(u_input):
  code = r'''

  blitz::Array<double,2> *p_u1 = &u1, *p_u2 = &u2;

  for (int iter = 0; iter < num_iters; ++iter) {
    if (iter != 0) {
      std::swap(p_u1, p_u2);
    }

    for (int i = 0; i < grid_size; ++i) {
      for (int j = 0; j < grid_size; ++j) {
        if (i == 0) {

          if (j == 0) {
            (*p_u2)(i,j) = (*p_u1)(i+1,j) - 1.0L/2.0L*(*p_u1)(i+2,j) + (*p_u1)(i,j+1) - 1.0L/2.0L*(*p_u1)(i,j+2);
          } else if (j == grid_size-1) {
            (*p_u2)(i,j) = (*p_u1)(i+1,j) - 1.0L/2.0L*(*p_u1)(i+2,j) + (*p_u1)(i,j-1) - 1.0L/2.0L*(*p_u1)(i,j-2);
          } else {
            (*p_u2)(i,j) = -2*(*p_u1)(i+1,j) + (*p_u1)(i+2,j) + (*p_u1)(i,j+1) + (*p_u1)(i,j-1);
          }

        } else if (i == grid_size-1) {

          if (j == 0) {
            (*p_u2)(i,j) = (*p_u1)(i,j+1) - 1.0L/2.0L*(*p_u1)(i,j+2) + (*p_u1)(i-1,j) - 1.0L/2.0L*(*p_u1)(i-2,j);
          } else if (j == grid_size-1) {
            (*p_u2)(i,j) = (*p_u1)(i,j-1) - 1.0L/2.0L*(*p_u1)(i,j-2) + (*p_u1)(i-1,j) - 1.0L/2.0L*(*p_u1)(i-2,j);
          } else {
            (*p_u2)(i,j) = (*p_u1)(i,j+1) + (*p_u1)(i,j-1) - 2*(*p_u1)(i-1,j) + (*p_u1)(i-2,j);
          }

        } else {

          if (j == 0) {
            (*p_u2)(i,j) = (*p_u1)(i+1,j) - 2*(*p_u1)(i,j+1) + (*p_u1)(i,j+2) + (*p_u1)(i-1,j);
          } else if (j == grid_size-1) {
            (*p_u2)(i,j) = (*p_u1)(i+1,j) - 2*(*p_u1)(i,j-1) + (*p_u1)(i,j-2) + (*p_u1)(i-1,j);
          } else {
            (*p_u2)(i,j) = (1.0L/4.0L)*(*p_u1)(i+1,j) + (1.0L/4.0L)*(*p_u1)(i,j+1) + (1.0L/4.0L)*(*p_u1)(i,j-1) + (1.0L/4.0L)*(*p_u1)(i-1,j);
          }

        }
      }
    }

    printf("iter %d done %d\n", iter, p_u2 == &u1);
  }

  return_val = p_u2 == &u1;
  '''

  u1 = u_input.astype(float).copy()
  u2 = u1.copy()

  num_iters = 1000

  assert u_input.shape[0] == u_input.shape[1]
  grid_size = u_input.shape[0]

  last_is_u1 = scipy.weave.inline(code, ['u1', 'u2', 'num_iters', 'grid_size'], type_converters=scipy.weave.converters.blitz)
  out = u1 if last_is_u1 else u2
  return out

  # print u_input
  # print out


def main():
  np.random.seed(0)

  h = sympy.Symbol('h')
  u = FieldVars('u', h, h)
  v = FieldVars('v', h, h)
  # print u.finite_diff_x()
  # print u.finite_diff_x(0)
  # print u.finite_diff_x(2)
  # print
  # print u.finite_diff_y()
  # print u.finite_diff_y(0)
  # print u.finite_diff_y(2)
  #
  # print u.finite_diff_y()

  # print sympy.simplify(u.dx())
  # print sympy.simplify(u.dy('c'))
  # print sympy.simplify(u.dy2('f'))
  # print sympy.simplify(u.dxdy())

  u.set_x_mode('c'); u.set_y_mode('c')
  v.set_x_mode('c'); v.set_y_mode('c')
  # e = sympy.simplify(u.dx2() + u.dy2())
  # print sympy.printing.ccode(e)
  # # from sympy.utilities.codegen import codegen
  # # print codegen(('f', e), 'C', 'test', header=False, empty=False)[0][1]
  # print sympy.latex(sympy.solve(u.dx2() + u.dy2() - sympy.Symbol('f'), u[0,0]))
  # print sympy.latex(sympy.Symbol(r'\tilde{w}')**2)

  # e = sympy.simplify(sympy.solve(u.dx2() + u.dy2(), u[0,0]))
  # print e

  def fn():
    return sympy.simplify(sympy.solve(u.dx2() + u.dy2(), u[0,0]))[0]

  Jacobi(fn, u)
  # return

  u_in = np.random.rand(100,100)
  u_out_0 = jacobi_laplace_equation(u_in)
  u_out = Jacobi(fn, u).run(u_in, 1000)
  print 'ok?', np.allclose(u_out_0, u_out)
  import matplotlib.pyplot as plt
  plt.figure(1)
  plt.imshow(u_in, cmap='gray')
  plt.figure(2)
  plt.imshow(u_out, cmap='gray')
  plt.show()

  return
  ############################















  ##################


  wtilde = sympy.Symbol(r'\tilde{w}_{i,j}')
  mu0 = sympy.Symbol(r'\tilde{\mu}^0_{i,j}')
  muu = sympy.Symbol(r'\tilde{\mu}^u_{i,j}')
  muv = sympy.Symbol(r'\tilde{\mu}^v_{i,j}')
  phi = sympy.Symbol(r'\phi_{i,j}')
  alpha = sympy.Symbol(r'\alpha') # const
  beta = sympy.Symbol(r'\beta') # const
  gamma = sympy.Symbol(r'\gamma') # const
  phi0 = sympy.Symbol(r'\phi^0_{i,j}')
  u0 = sympy.Symbol(r'u^0_{i,j}')
  v0 = sympy.Symbol(r'v^0_{i,j}')

  u.set_mode('c', 'c')
  v.set_mode('c', 'c')
  expr = -2*wtilde*muu*(phi - mu0 - muu*u[0,0] - muv*v[0,0]) + 2*beta*u[0,0] + 2*gamma*(u[0,0] - u0) - alpha*(8*u.dx2() + 4*(v.dxdy() + u.dy2()))
  print sympy.latex(sympy.simplify(sympy.solve(expr, u[0,0])), fold_short_frac=True)
  print
  expr = -2*wtilde*muv*(phi - mu0 - muu*u[0,0] - muv*v[0,0]) + 2*beta*v[0,0] + 2*gamma*(v[0,0] - v0) - alpha*(8*v.dy2() + 4*(u.dxdy() + v.dx2()))
  print sympy.latex(sympy.simplify(sympy.solve(expr, v[0,0])), fold_short_frac=True)
if __name__ == '__main__':
  main()
