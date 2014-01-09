import numpy as np
import ctimbpy

opt = ctimbpy.Optimizer()
x, y = opt.add_vars(['x', 'y'])

#cost_x = ctimbpy.SimpleCost(x, 3, 'cost_x')
#import IPython; IPython.embed()
opt.add_cost(ctimbpy.SimpleCost(x, 3, 'cost_x'))
opt.add_cost(ctimbpy.SimpleCost(y, -1, 'cost_y'))

print 'num vars', opt.num_vars()

result = opt.optimize(np.array([3, 5]))
print 'result', result.x
print 'cost', result.cost
