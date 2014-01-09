import numpy as np
import ctimbpy

opt = ctimbpy.Optimizer()
x, y = opt.add_vars(['x', 'y'])
opt.params().max_iter = 4
opt.params().check_linearizations = False

opt.add_cost(ctimbpy.ExampleCost(x, 3, 'cost_x'))
opt.add_cost(ctimbpy.ExampleCost(y, -1, 'cost_y'))

print 'num vars', opt.num_vars()

result = opt.optimize(np.array([3, 5]))
print 'result', result.x
print 'cost', result.cost
