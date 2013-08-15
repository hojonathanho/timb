import unittest
from grid import *

class Tests(unittest.TestCase):

  def test_grid_interp(self):
    n, m = 5, 4
    data = np.random.rand(n, m)
    u = np.transpose(np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, m-1, m)))
    g2 = grid_interp(data, u.reshape((-1, 2))).reshape((n, m))
    self.assertTrue(np.allclose(data, g2))
    for j in range(1,m):
      g3 = grid_interp(data, (u + [0,j]).reshape((-1, 2))).reshape((n, m))
      self.assertTrue(np.allclose(data[:,j:], g3[:,:-j]))
      self.assertTrue(np.allclose(g3[:,-1], data[:,-1]))

    vector_data = np.random.rand(n, m, 3)
    g4 = grid_interp(vector_data, u.reshape((-1, 2))).reshape((n, m, 3))
    self.assertTrue(np.allclose(vector_data, g4))

    matrix_data = np.random.rand(n, m, 2, 3)
    g5 = grid_interp(matrix_data, u.reshape((-1, 2))).reshape((n, m, 2, 3))
    self.assertTrue(np.allclose(matrix_data, g5))

  def test_grid_interp_grad(self):
    n, m = 10, 15
    data = np.random.rand(n, m)
    u = np.transpose(np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, m-1, m))).reshape((-1,2))
    g1 = grid_interp_grad_nd(data, u)
    g2 = grid_interp_grad(data, u)
    self.assertTrue(np.allclose(g1, g2))

    u2 = np.random.rand(30, 2)
    u2[:,0] *= n-1; u[:,1] *= m-1
    g3 = grid_interp_grad_nd(data, u2)
    g4 = grid_interp_grad(data, u2)
    self.assertTrue(np.allclose(g3, g4))

  def test_func_eval(self):
    data = np.random.rand(4, 5)
    f = SpatialFunction(-1, 1,  6, 7, data)
    coords = np.transpose(np.meshgrid(np.linspace(-1, 1, 4), np.linspace(6, 7, 5)))
    vals = f.eval_xys(coords.reshape((-1,2))).reshape(data.shape)
    self.assertTrue(np.allclose(vals, data))

  def test_flow(self):
    data = np.random.rand(4, 5)
    f = SpatialFunction(-1, 1,  6, 8, data)

    zero_flow = SpatialFunction(-1, 1, 6, 8, np.zeros(data.shape + (2,)))
    f2 = f.flow(zero_flow)
    self.assertTrue(np.allclose(f2.data(), data))

    x_flow_data = np.zeros(data.shape + (2,))
    x_flow_data[:,:,0] = 2./3.
    x_flow = SpatialFunction(-1, 1, 6, 8, x_flow_data)
    f3 = f.flow(x_flow)
    self.assertTrue(np.allclose(f3.data()[1:,:], data[:-1,:]))
    self.assertTrue(np.allclose(f3.data()[0,:], data[0,:]))

  def test_num_jac(self):
    '''test jacobians on a function f(x) = [g_1(x), g_2(x), g_3(x)] where g_i are scaled rosenbrock functions'''
    def rosen(x):
      return (100.0*(x[:,1:]-x[:,:-1]**2.0)**2.0 + (1-x[:,:-1])**2.0).sum(axis=1)
    def rosen_der(x):
      xm = x[:,1:-1]
      xm_m1 = x[:,:-2]
      xm_p1 = x[:,2:]
      der = np.zeros_like(x)
      der[:,1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
      der[:,0] = -400*x[:,0]*(x[:,1]-x[:,0]**2) - 2*(1-x[:,0])
      der[:,-1] = 200*(x[:,-1]-x[:,-2]**2)
      return der

    ndim = 3
    n, m = 50, 60
    xmin, xmax, ymin, ymax = -1, 1, 6, 8
    u = np.transpose(np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, m))).reshape((-1,2))
    xs, ys = u[:,0], u[:,1]
    data = rosen(u).reshape((n, m)); data = np.dstack([(k+1.)*data for k in range(ndim)])

    f = SpatialFunction(xmin, xmax, ymin, ymax, data)
    jacs = f.num_jac_direct(xs, ys)
    jacs2 = f.num_jac(xs, ys)
    self.assertTrue(np.allclose(jacs, jacs2))

    true_jac_base = rosen_der(u)
    true_jacs = np.empty_like(jacs)
    for k in range(ndim):
      true_jacs[:,k,:] = (k+1.)*true_jac_base
    self.assertTrue(np.absolute(true_jacs - jacs).max()/true_jacs.ptp() < .01)

    # second derivatives
    jac_fn = SpatialFunction(xmin, xmax, ymin, ymax, jacs.reshape((n, m, ndim, 2)))
    self.assertTrue(np.allclose(jac_fn.num_jac_direct(xs, ys), jac_fn.num_jac(xs, ys)))


def run_tests():
  suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
  unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
  unittest.main()
