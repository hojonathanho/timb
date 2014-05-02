import numpy as np
import random
import sympy
import scipy.weave
import re
from collections import defaultdict

def random_str(length=10):
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



def n_to_offset(prefix, n):
  if n == 0: return prefix
  if n > 0: return '%s+%d' % (prefix, n)
  return '%s-%d' % (prefix, -n)


def make_var_symbols(prefix, order):
  assert order >= 1
  size = 2*order + 1
  a = np.empty((size,size), dtype=object)
  name2pos = {}
  for i in range(size):
    for j in range(size):
      # a[i,j] = sympy.Symbol('%s_{%s,%s}' % (name, n_to_offset('i', i-order), n_to_offset('j', j-order)))
      name = '%s%03d%03d%s' % (prefix.replace('_',''), i-order, j-order, random_str())
      assert name not in name2pos
      name2pos[name] = (i,j)
      a[i,j] = sympy.Symbol(name)
  return a, name2pos


class FieldVars(object):
  def __init__(self, name, hx, hy, order=2, constant=False, latex_name=None):
    self.name = name
    self.order = 0 if constant else order
    self.latex_name = name if latex_name is None else latex_name # todo
    self.constant = constant
    if constant:
      self.u = np.empty((1,1), dtype=object)
      var_name = name.replace('_','') + random_str()
      self.u[0,0] = sympy.Symbol(var_name)
      self.internalname2pos = { var_name: (0,0) }
    else:
      self.u, self.internalname2pos = make_var_symbols(name, order)
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

  def replace_var_names(self, s, style='c', i_offset=0, j_offset=0):
    if style == 'c':
      name = self.name
      fmt = '%s(%s,%s)'
    elif style == 'latex':
      name = self.latex_name
      fmt = '%s_{%s,%s}'
    else:
      raise NotImplementedError

    if self.constant:
      assert len(self.internalname2pos) == 1
      return s.replace(list(self.internalname2pos)[0], name)

    for n in self.internalname2pos:
      i, j = self.internalname2pos[n]
      s = s.replace(n, fmt % (name, n_to_offset('i', i-self.order+i_offset), n_to_offset('j', j-self.order+j_offset)))
    return s


def gen_jacobi():

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


  def replace_names(s, style='c', i_offset=0, j_offset=0):
    for a in all_vars:
      s = a.replace_var_names(s, style, i_offset, j_offset)
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


def gen_asdf():
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

  def replace_names(s, style='c', i_offset=0, j_offset=0):
    for a in all_vars:
      s = a.replace_var_names(s, style, i_offset, j_offset)
    return s

  def set_modes(m):
    for a in all_vars:
      a.set_mode(m[0], m[1])

  def gen_cost_derivs():
    cost = 0

    range_min, range_max = -2, 2

    for i in range(range_min, range_max+1):
      for j in range(range_min, range_max+1):
        # observation cost
        cost += wz[i,j] * (z[i,j] - phi[i,j])**2

        # quadraticized flow agreement cost
        cost += wtilde[i,j] * (phi[i,j] - mu0[i,j] - muu[i,j]*u[i,j] - muv[i,j]*v[i,j])**2

        # strain cost
        if i != range_max:
          du_dx = (u[i+1,j] - u[i,j])/h
          dv_dx = (v[i+1,j] - v[i,j])/h
        else:
          du_dx = (u[i,j] - u[i-1,j])/h
          dv_dx = (v[i,j] - v[i-1,j])/h

        if j != range_max:
          du_dy = (u[i,j+1] - u[i,j])/h
          dv_dy = (v[i,j+1] - v[i,j])/h
        else:
          du_dy = (u[i,j] - u[i,j-1])/h
          dv_dy = (v[i,j] - v[i,j-1])/h

        cost += alpha() * (4*du_dx**2 + 2*(dv_dx + du_dy)**2 + 4*dv_dy**2)

        # flow norm cost
        cost += beta() * (u[i,j]**2 + v[i,j]**2)

        # soft trust region
        cost += gamma() * ((phi[i,j] - phi0[i,j])**2 + (u[i,j] - u0[i,j])**2 + (v[i,j] - v0[i,j])**2)


    derivs = { 'phi': {}, 'u': {}, 'v': {} }
    for i in range(range_min, range_max+1):
      for j in range(range_min, range_max+1):
        print i,j
        derivs['phi'][(i,j)] = sympy.simplify(sympy.diff(cost, phi[i,j]))
        derivs['u'][(i,j)] = sympy.simplify(sympy.diff(cost, u[i,j]))
        derivs['v'][(i,j)] = sympy.simplify(sympy.diff(cost, v[i,j]))

    return cost, derivs


  cost, derivs = gen_cost_derivs()

  for i in range(-2, 3):
    # dcost_du = sympy.simplify(sympy.diff(cost, u[i,0]))
    print replace_names(sympy.printing.ccode(derivs['u'][(i,0)]), 'c', -i, 0)
    print

  # dcost_du = sympy.simplify(sympy.diff(cost, u[2,0])/2)
  # print replace_names(sympy.printing.ccode(dcost_du), 'c')



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














def main():
  np.random.seed(0)
  random.seed(0)
  gen_asdf()

if __name__ == '__main__':
  main()
