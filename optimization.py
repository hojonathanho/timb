import numpy as np
import scipy.optimize as sio

# def bt_line_sesarch(f, x, dx, alpha=, beta=):
#   t = 1.
#   f0 = f(x)
#   g0 = f_grad(x)
#   while f(x + t*dx) > f0 + alpha*t*g0.dot(dx):
#     t *= beta
#   return t

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
