import numpy as np
import ctimbpy

opt = ctimbpy.Optimizer()

NX = 5
NY = 10

gp = ctimbpy.GridParams(-1, 1, -1, 1, NX, NY)

phi_vars = ctimbpy.make_var_field(opt, 'phi', gp)
u_x_vars = ctimbpy.make_var_field(opt, 'u_x', gp)
u_y_vars = ctimbpy.make_var_field(opt, 'u_y', gp)

strain_cost = ctimbpy.FlowNormCost(u_x_vars, u_y_vars)
opt.add_cost(strain_cost)

rigidity_cost = ctimbpy.FlowRigidityCost(u_x_vars, u_y_vars)
opt.add_cost(rigidity_cost)

print 'num vars', opt.num_vars()

init_phi_vals = np.zeros((NX, NY))
init_u_x_vals = np.zeros((NX, NY))
init_u_y_vals = np.zeros((NX, NY))

def to_vec(phi, u_x, u_y):
  return np.r_[phi.ravel(), u_x.ravel(), u_y.ravel()]

result = opt.optimize(to_vec(init_phi_vals, init_u_x_vals, init_u_y_vals))
print 'result', result.x
print 'cost', result.cost
