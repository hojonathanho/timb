import numpy as np
import theano.tensor as T
from theano import pp
import theano

#x = T.dvector('x')
#y = x ** 2
#print 'x', pp(x)
#print 'y', pp(y)#

#print 'grad'
#gy = theano.gradient.jacobian(y, x)

#f = theano.function([x], gy)
#pp(f.maker.fgraph.outputs[0])


#x = T.scalar('x')
#y = theano.ifelse.ifelse(T.lt(x, 0), -x, x)
#piecewise_f = theano.function([x], y)
#print pp(piecewise_f.maker.fgraph.outputs[0])

#piecewise_f_grad = theano.function([x], T.grad(y, x))
#print pp(piecewise_f_grad.maker.fgraph.outputs[0])
#print type(x), x.shape[2]


def theano_grad(u, eps_x=1, eps_y=1):
    '''Gradients of scalar field
    Input: u (NxM)
    Output: gradient field (NxMx2)
    '''
    gx_left = (u[1,:][None,:] - u[0,:][None,:]) / eps_x
    gx_middle = (u[2:,:] - u[:-2,:]) / (2*eps_x)
    gx_right = (u[-1,:][None,:] - u[-2,:][None,:]) / eps_x
    gx = T.concatenate([gx_left, gx_middle, gx_right], axis=0)

    gy_left = (u[:,1][:,None] - u[:,0][:,None]) / eps_y
    gy_middle = (u[:,2:] - u[:,:-2]) / (2*eps_y)
    gy_right = (u[:,-1][:,None] - u[:,-2][:,None]) / eps_y
    gy = T.concatenate([gy_left, gy_middle, gy_right], axis=1)
    
    g = T.concatenate([gx[:,:,None], gy[:,:,None]], axis=2)
    return g

def theano_jac(u, eps_x=1, eps_y=1):
    J = T.concatenate([theano_grad(u[:,:,0], eps_x, eps_y)[:,:,:,None], theano_grad(u[:,:,1], eps_x, eps_y)[:,:,:,None]], axis=3)
    return J.dimshuffle(0, 1, 3, 2)

def np_jac(u):
    J_np = np.empty((u.shape[0], u.shape[1], 2, 2))
    J_np[:,:,0,:] = np.dstack(np.gradient(u[:,:,0]))
    J_np[:,:,1,:] = np.dstack(np.gradient(u[:,:,1]))
    return J_np

u = T.dtensor3('u')

cost_u_norm = (u**2).sum()
print 'cost_u_norm', pp(cost_u_norm)



#test_u = np.random.rand(2, 3, 2)
#print 'yeah'
#print theano.function([u], T.grad((theano_grad(u[:,:,0])**2).sum(), u))(test_u)



J_u = theano_jac(u)
J_u_func = theano.function([u], J_u)

print pp(u)
test_u = np.random.rand(2, 3, 2)
print 'arr', test_u
J_theano = J_u_func(test_u)
print 'grad', J_theano
J_np = np_jac(test_u)
print 'np grad', J_np
print 'ok?', np.allclose(J_theano, J_np)

print 'rigidity cost'
cost_rigidity = (J_u**2).sum()
cost_rigidity_func = theano.function([u], cost_rigidity)
print '= ', cost_rigidity_func(test_u)
print 'np rigidity cost', (J_np**2).sum()

print 'rigidity gradient'
cost_rigidity_grad = T.grad(cost_rigidity, u)
print pp(cost_rigidity_grad)
theano_gval = theano.function([u], cost_rigidity_grad)(test_u)
print theano_gval

eps = 1e-5
fn = lambda x: (np_jac(x)**2).sum()
np_gval = np.empty_like(test_u)
for i in range(test_u.shape[0]):
  for j in range(test_u.shape[1]):
    for k in range(test_u.shape[2]):
      orig = test_u[i,j,k]
      test_u[i,j,k] = orig + eps
      y2 = fn(test_u)
      test_u[i,j,k] = orig - eps
      y1 = fn(test_u)
      test_u[i,j,k] = orig
      np_gval[i,j,k] = (y2-y1)/(2.*eps)
print 'np grad'
print np_gval

print 'allclose?', np.allclose(theano_gval, np_gval)

