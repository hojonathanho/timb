import numpy as np
import random
import sympy
import scipy.weave
import re
from collections import defaultdict

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
      a[i,j] = sympy.Symbol('%s_{%s,%s}' % (name, n_to_offset('i', i-order), n_to_offset('j', j-order)))
  return a


class FieldVars(object):
  def __init__(self, name, hx, hy, order=2, constant=False, latex_name=None):
    self.name = name
    self.order = 0 if constant else order
    self.internal_name = name.replace('_','') + random_str(10)
    self.latex_name = name if latex_name is None else latex_name # todo
    if constant:
      self.u = np.empty((1,1), dtype=object)
      self.u[0,0] = sympy.Symbol(self.internal_name)
    else:
      self.u = make_var_symbols(self.internal_name, order)
    self.hx, self.hy = hx, hy

    self.mode_x, self.mode_y = 'c', 'c'

  def __getitem__(self, idx):
    assert len(idx) == 2
    return self.u[idx[0]+self.order, idx[1]+self.order]

  def __call__(self):
    return self[0,0]

  def set_x_mode(self, mode_x):
    self.mode_x = mode_x

  def set_y_mode(self, mode_y):
    self.mode_y = mode_y

  def set_mode(self, mode_x, mode_y):
    self.mode_x, self.mode_y = mode_x, mode_y


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

  def replace_var_names(self, s, style='c'):
    if style == 'c':
      s = s.replace(self.internal_name, self.name)
      s = s.replace('_{', '(')
      s = s.replace('}', ')')
      return s
    elif style == 'latex':
      return s.replace(self.internal_name, self.latex_name)
    else:
      raise NotImplementedError


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

        }} else if (i == {grid_size}-1) {{

          if (j == 0) {{
            (*{var_ptr_name_2})(i,j) = {exprs[bf]};
          }} else if (j == {grid_size}-1) {{
            (*{var_ptr_name_2})(i,j) = {exprs[bb]};
          }} else {{
            (*{var_ptr_name_2})(i,j) = {exprs[bc]};
          }}

        }} else {{

          if (j == 0) {{
            (*{var_ptr_name_2})(i,j) = {exprs[cf]};
          }} else if (j == {grid_size}-1) {{
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


# def jacobi_laplace_equation(u_input):
#   code = r'''

#   blitz::Array<double,2> *p_u1 = &u1, *p_u2 = &u2;

#   for (int iter = 0; iter < num_iters; ++iter) {
#     if (iter != 0) {
#       std::swap(p_u1, p_u2);
#     }

#     for (int i = 0; i < grid_size; ++i) {
#       for (int j = 0; j < grid_size; ++j) {
#         if (i == 0) {

#           if (j == 0) {
#             (*p_u2)(i,j) = (*p_u1)(i+1,j) - 1.0L/2.0L*(*p_u1)(i+2,j) + (*p_u1)(i,j+1) - 1.0L/2.0L*(*p_u1)(i,j+2);
#           } else if (j == grid_size-1) {
#             (*p_u2)(i,j) = (*p_u1)(i+1,j) - 1.0L/2.0L*(*p_u1)(i+2,j) + (*p_u1)(i,j-1) - 1.0L/2.0L*(*p_u1)(i,j-2);
#           } else {
#             (*p_u2)(i,j) = -2*(*p_u1)(i+1,j) + (*p_u1)(i+2,j) + (*p_u1)(i,j+1) + (*p_u1)(i,j-1);
#           }

#         } else if (i == grid_size-1) {

#           if (j == 0) {
#             (*p_u2)(i,j) = (*p_u1)(i,j+1) - 1.0L/2.0L*(*p_u1)(i,j+2) + (*p_u1)(i-1,j) - 1.0L/2.0L*(*p_u1)(i-2,j);
#           } else if (j == grid_size-1) {
#             (*p_u2)(i,j) = (*p_u1)(i,j-1) - 1.0L/2.0L*(*p_u1)(i,j-2) + (*p_u1)(i-1,j) - 1.0L/2.0L*(*p_u1)(i-2,j);
#           } else {
#             (*p_u2)(i,j) = (*p_u1)(i,j+1) + (*p_u1)(i,j-1) - 2*(*p_u1)(i-1,j) + (*p_u1)(i-2,j);
#           }

#         } else {

#           if (j == 0) {
#             (*p_u2)(i,j) = (*p_u1)(i+1,j) - 2*(*p_u1)(i,j+1) + (*p_u1)(i,j+2) + (*p_u1)(i-1,j);
#           } else if (j == grid_size-1) {
#             (*p_u2)(i,j) = (*p_u1)(i+1,j) - 2*(*p_u1)(i,j-1) + (*p_u1)(i,j-2) + (*p_u1)(i-1,j);
#           } else {
#             (*p_u2)(i,j) = (1.0L/4.0L)*(*p_u1)(i+1,j) + (1.0L/4.0L)*(*p_u1)(i,j+1) + (1.0L/4.0L)*(*p_u1)(i,j-1) + (1.0L/4.0L)*(*p_u1)(i-1,j);
#           }

#         }
#       }
#     }

#     printf("iter %d done %d\n", iter, p_u2 == &u1);
#   }

#   return_val = p_u2 == &u1;
#   '''

#   u1 = u_input.astype(float).copy()
#   u2 = u1.copy()

#   num_iters = 1000

#   assert u_input.shape[0] == u_input.shape[1]
#   grid_size = u_input.shape[0]

#   last_is_u1 = scipy.weave.inline(code, ['u1', 'u2', 'num_iters', 'grid_size'], type_converters=scipy.weave.converters.blitz)
#   out = u1 if last_is_u1 else u2
#   return out





def solve(num_iters, h, u_init, v_init, phi_init, z, w_z, mu_0, mu_u, mu_v, wtilde, alpha, beta, gamma, phi_0, u_0, v_0):
  code = r'''
  typedef blitz::Array<double, 2> Array;

  for (int iter = 0; iter < num_iters; ++iter) {
    Array& u = (iter % 2 == 0) ? u_input : mirror_u_input;
    Array& v = (iter % 2 == 0) ? v_input : mirror_v_input;
    Array& phi = (iter % 2 == 0) ? phi_input : mirror_phi_input;

    Array& mirror_u = (iter % 2 == 1) ? u_input : mirror_u_input;
    Array& mirror_v = (iter % 2 == 1) ? v_input : mirror_v_input;
    Array& mirror_phi = (iter % 2 == 1) ? phi_input : mirror_phi_input;

    for (int i = 0; i < grid_size; ++i) {
      for (int j = 0; j < grid_size; ++j) {
        if (i == 0) {

          if (j == 0) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) - 4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) + 2*alpha*v(i+1,j+1) - 2*alpha*v(i+1,j) - 2*alpha*v(i,j+1) + 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            mirror_v(i,j) = (2*alpha*u(i+1,j+1) - 2*alpha*u(i+1,j) - 2*alpha*u(i,j+1) + 2*alpha*u(i,j) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else if (j == grid_size-1) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) - 4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) - 2*alpha*v(i+1,j-1) + 2*alpha*v(i+1,j) + 2*alpha*v(i,j-1) - 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            mirror_v(i,j) = (-2*alpha*u(i+1,j-1) + 2*alpha*u(i+1,j) + 2*alpha*u(i,j-1) - 2*alpha*u(i,j) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) + 2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) + alpha*v(i+1,j+1) - alpha*v(i+1,j-1) - alpha*v(i,j+1) + alpha*v(i,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_u(i,j)*mu_u(i,j))*wtilde(i,j)));

            mirror_v(i,j) = (alpha*u(i+1,j+1) - alpha*u(i+1,j-1) - alpha*u(i,j+1) + alpha*u(i,j-1) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          }

        } else if (i == grid_size-1) {

          if (j == 0) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (-4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) + 2*alpha*v(i,j+1) - 2*alpha*v(i,j) - 2*alpha*v(i-1,j+1) + 2*alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            mirror_v(i,j) = (2*alpha*u(i,j+1) - 2*alpha*u(i,j) - 2*alpha*u(i-1,j+1) + 2*alpha*u(i-1,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else if (j == grid_size-1) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (-4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) - 2*alpha*v(i,j-1) + 2*alpha*v(i,j) + 2*alpha*v(i-1,j-1) - 2*alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            mirror_v(i,j) = (-2*alpha*u(i,j-1) + 2*alpha*u(i,j) + 2*alpha*u(i-1,j-1) - 2*alpha*u(i-1,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) + alpha*v(i,j+1) - alpha*v(i,j-1) - alpha*v(i-1,j+1) + alpha*v(i-1,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_u(i,j)*mu_u(i,j))*wtilde(i,j)));

            mirror_v(i,j) = (alpha*u(i,j+1) - alpha*u(i,j-1) - alpha*u(i-1,j+1) + alpha*u(i-1,j-1) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          }

        } else {

          if (j == 0) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (4*alpha*u(i+1,j) - 4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) + 4*alpha*u(i-1,j) + alpha*v(i+1,j+1) - alpha*v(i+1,j) - alpha*v(i-1,j+1) + alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            mirror_v(i,j) = (alpha*u(i+1,j+1) - alpha*u(i+1,j) - alpha*u(i-1,j+1) + alpha*u(i-1,j) + 2*alpha*v(i+1,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_v(i,j)*mu_v(i,j))*wtilde(i,j)));

          } else if (j == grid_size-1) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (4*alpha*u(i+1,j) - 4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) + 4*alpha*u(i-1,j) - alpha*v(i+1,j-1) + alpha*v(i+1,j) + alpha*v(i-1,j-1) - alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            mirror_v(i,j) = (-alpha*u(i+1,j-1) + alpha*u(i+1,j) + alpha*u(i-1,j-1) - alpha*u(i-1,j) + 2*alpha*v(i+1,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_v(i,j)*mu_v(i,j))*wtilde(i,j)));

          } else {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (4*alpha*u(i+1,j) + 2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) + 4*alpha*u(i-1,j) + (1.0L/2.0L)*alpha*v(i+1,j+1) - 1.0L/2.0L*alpha*v(i+1,j-1) - 1.0L/2.0L*alpha*v(i-1,j+1) + (1.0L/2.0L)*alpha*v(i-1,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            mirror_v(i,j) = ((1.0L/2.0L)*alpha*u(i+1,j+1) - 1.0L/2.0L*alpha*u(i+1,j-1) - 1.0L/2.0L*alpha*u(i-1,j+1) + (1.0L/2.0L)*alpha*u(i-1,j-1) + 2*alpha*v(i+1,j) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          }

        }
      }
    } // end iteration over grid

    // printf("iter %d done\n", iter);
  }
  '''

  assert u_init.shape[0] == u_init.shape[1]
  assert u_init.shape == v_init.shape == phi_init.shape == z.shape == w_z.shape == mu_0.shape == mu_u.shape == mu_v.shape == wtilde.shape == phi_0.shape == u_0.shape == v_0.shape

  grid_size = u_init.shape[0]

  u = u_init.astype(float).copy()
  v = v_init.astype(float).copy()
  phi = phi_init.astype(float).copy()

  mirror_u, mirror_v, mirror_phi = u.copy(), v.copy(), phi.copy()

  local_dict = {
    'u_input': u,
    'v_input': v,
    'phi_input': phi,
    'mirror_u_input': mirror_u,
    'mirror_v_input': mirror_v,
    'mirror_phi_input': mirror_phi,

    'z': z,
    'w_z': w_z,
    'mu_0': mu_0,
    'mu_u': mu_u,
    'mu_v': mu_v,
    'wtilde': wtilde,
    'alpha': alpha,
    'beta': beta,
    'gamma': gamma,
    'phi_0': phi_0,
    'u_0': u_0,
    'v_0': v_0,

    'h': h,
    'grid_size': grid_size,
    'num_iters': num_iters
  }

  scipy.weave.inline(
    code,
    local_dict.keys(),
    local_dict,
    type_converters=scipy.weave.converters.blitz
  )

  return (u, v, phi) if (num_iters % 2 == 0) else (mirror_u, mirror_v, mirror_phi)


def main():
  np.random.seed(0)


  size = 100
  u_init = np.random.rand(size,size)
  v_init = np.random.rand(size,size)
  phi_init = np.random.rand(size,size)
  z = np.random.rand(size,size)
  w_z = np.random.rand(size,size)
  mu_0 = np.random.rand(size,size)
  mu_u = np.random.rand(size,size)
  mu_v = np.random.rand(size,size)
  wtilde = np.random.rand(size,size)
  alpha = 10
  beta = 20
  gamma = 30
  phi_0 = np.random.rand(size,size)
  u_0 = np.random.rand(size,size)
  v_0 = np.random.rand(size,size)
  h = 1.

  import time
  t_start = time.time()
  for i in range(10):
    for j in range(10):
      solve(10, h, u_init, v_init, phi_init, z, w_z, mu_0, mu_u, mu_v, wtilde, alpha, beta, gamma, phi_0, u_0, v_0)
  t_end = time.time()
  print 'time:', t_end - t_start
  return



# u.set_x_mode('c'); u.set_y_mode('c')
# v.set_x_mode('c'); v.set_y_mode('c')
# def fn():
#   return sympy.simplify(sympy.solve(u.dx2() + u.dy2(), u[0,0]))[0]

# Jacobi(fn, u)
# # return

# u_in = np.random.rand(100,100)
# u_out_0 = jacobi_laplace_equation(u_in)
# u_out = Jacobi(fn, u).run(u_in, 1000)
# print 'ok?', np.allclose(u_out_0, u_out)
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.imshow(u_in, cmap='gray')
# plt.figure(2)
# plt.imshow(u_out, cmap='gray')
# plt.show()

  ############################


  ##################

  all_vars = []
  def add_vars(*args, **kwargs):
    fv = FieldVars(*args, **kwargs)
    all_vars.append(fv)
    return fv

  # grid spacing
  h = sympy.Symbol('h')

  # variables to solve for
  u = add_vars('u', h, h, latex_name='u')
  v = add_vars('v', h, h, latex_name='v')
  phi = add_vars('phi', h, h, latex_name=r'\phi')

  # observation weights and values
  z = add_vars('z', h, h, latex_name=r'z')
  wz = add_vars('w_z', h, h, latex_name=r'w^z')

  # flowed weights and phi values
  mu0 = add_vars('mu_0', h, h, latex_name=r'\tilde{\mu}^0')
  muu = add_vars('mu_u', h, h, latex_name=r'\tilde{\mu}^u')
  muv = add_vars('mu_v', h, h, latex_name=r'\tilde{\mu}^v')
  wtilde = add_vars('wtilde', h, h, latex_name=r'\tilde{w}')

  # cost coefficients (alpha: strain, beta: norm)
  alpha = add_vars('alpha', h, h, latex_name=r'\alpha', constant=True)
  beta = add_vars('beta', h, h, latex_name=r'\beta', constant=True)

  # trust region coefficient
  gamma = add_vars('gamma', h, h, latex_name=r'\gamma', constant=True)
  # trust region center
  phi0 = add_vars('phi_0', h, h, latex_name=r'\phi^0')
  u0 = add_vars('u_0', h, h, latex_name=r'u^0')
  v0 = add_vars('v_0', h, h, latex_name=r'v^0')

  # all_vars = [u, v, wtilde, mu0, muu, muv, phi, alpha, beta, gamma, phi0, u0, v0]

  def replace_names(s, style='c'):
    for a in all_vars:
      s = a.replace_var_names(s, style)
    return s

  def set_modes(m):
    for a in all_vars:
      a.set_mode(m[0], m[1])

  def print_eqns(expr, var):
    # print 'Update equations for', var.name
    update_expr = sympy.simplify(sympy.solve(expr, var()))[0]
    # print 'C:'
    out = replace_names(str(var()), 'c') + ' = ' + replace_names(sympy.printing.ccode(update_expr), 'c') + ';'
    # get rid of pow(x, 2) and replace with x*x
    p = re.compile('pow\((.+?), 2\)')
    return p.sub(r'(\1*\1)', out)
    # print '\nLatex:\n', replace_names(sympy.latex(update_expr, fold_short_frac=True), 'latex')
    # print '\n'

  def print_var_names():
    print ', '.join(v.name for v in all_vars)

  print_var_names()

  code_dict = defaultdict(list)
  for modes in ['ff', 'fb', 'fc', 'bf', 'bb', 'bc', 'cf', 'cb', 'cc']:
    print '========== %s ==========' % modes
    set_modes(modes)

    expr = 2*wz()*(phi() - z()) + 2*wtilde()*(phi() - mu0() - muu()*u() - muv()*v()) + 2*gamma()*(phi() - phi0())
    code_dict[modes].append(print_eqns(expr, phi))

    expr = -2*wtilde()*muu()*(phi() - mu0() - muu()*u() - muv()*v()) + 2*beta()*u() + 2*gamma()*(u() - u0()) - alpha()*(8*u.dx2() + 4*(v.dxdy() + u.dy2()))
    code_dict[modes].append(print_eqns(expr, u))

    expr = -2*wtilde()*muv()*(phi() - mu0() - muu()*u() - muv()*v()) + 2*beta()*v() + 2*gamma()*(v() - v0()) - alpha()*(8*v.dy2() + 4*(u.dxdy() + v.dx2()))
    code_dict[modes].append(print_eqns(expr, v))

    print code_dict[modes]

  code_template = '''
for (int i = 0; i < grid_size; ++i) {{
  for (int j = 0; j < grid_size; ++j) {{
    if (i == 0) {{

      if (j == 0) {{

        {exprs[ff]}

      }} else if (j == grid_size-1) {{

        {exprs[fb]}

      }} else {{

        {exprs[fc]}

      }}

    }} else if (i == grid_size-1) {{

      if (j == 0) {{

        {exprs[bf]}

      }} else if (j == grid_size-1) {{

        {exprs[bb]}

      }} else {{

        {exprs[bc]}

      }}

    }} else {{

      if (j == 0) {{

        {exprs[cf]}

      }} else if (j == grid_size-1) {{

        {exprs[cb]}

      }} else {{

        {exprs[cc]}

      }}

    }}
  }}
}}
  '''
  print code_template.format(exprs=dict((k, '\n\n        '.join(v)) for (k, v) in code_dict.iteritems()))

if __name__ == '__main__':
  main()
