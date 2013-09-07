import numpy as np
import sqp_problem
import matplotlib.pyplot as plt
import matplotlib.cm as cm

SIZE = 30
WORLD_MIN = (0., 0.)
WORLD_MAX = (SIZE-1., SIZE-1.)
sqp_problem.Config.set(SIZE, SIZE, WORLD_MIN, WORLD_MAX)

def apply_flow(phi, u):
  surf = sqp_problem.make_interp(phi)
  out = np.empty_like(phi)
  for i in range(out.shape[0]):
    for j in range(out.shape[1]):
      out[i,j] = surf.eval_xys(surf.to_xys([i,j])[0] - u[i,j,:])
  return out

def apply_flow_forwards(fig, N, phi, u, x=None, y=None, z=None):
  # x goes to x + u(x)
  # so we want out(x+u(x)) == phi(x)
  surf = sqp_problem.make_interp(phi)
  pts = np.empty((phi.shape + (2,)))
  if x is None or y is None or z is None:
    x, y, z = [], [], []
    for i in range(phi.shape[0]):
      for j in range(phi.shape[1]):
        p = surf.to_xys([i,j])[0] + u[i,j,:]
        x.append(p[0])
        y.append(p[1])
        z.append(phi[i,j])

  ax = fig.add_subplot(N, aspect='equal')
  #pts = pts.reshape((-1,2))
  #ax.scatter(pts[:,0], pts[:,1], c=phi.reshape(-1), cmap='Greys_r')
  ax.scatter(x, y, c=z, cmap='Greys_r')


def plot_phi(fig, N, phi):
  x = np.linspace(WORLD_MIN[0], WORLD_MAX[0], phi.nx)
  y = np.linspace(WORLD_MIN[1], WORLD_MAX[1], phi.ny)
  # X, Y = np.meshgrid(x, y)
  pts = np.c_[np.repeat(x, phi.ny), np.tile(y, phi.nx)]
  Z = phi.eval_xys(pts).reshape((phi.nx, phi.ny)).T
  Z = np.flipud(Z)
  ax = fig.add_subplot(N, aspect='equal')
  ax.imshow(Z, cmap=cm.Greys_r).set_interpolation('nearest')
  ax.contour(X,Y,Z,levels=[0,.1,.5])
  return ax

import cPickle
with open('/tmp/dump.pkl', 'r') as f:
  data = cPickle.load(f)

with open('/tmp/init_phi.pkl', 'r') as f:
  init_phi = cPickle.load(f)

u = data[1]
#u = -u

#u.fill(0)
#u[:,:,0] = -1
#u[:,:,1] = .5


def rot(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s], [s, c]], dtype=float)

x = np.linspace(WORLD_MIN[0], WORLD_MAX[0], u.shape[0]).astype(float)
y = np.linspace(WORLD_MIN[1], WORLD_MAX[1], u.shape[1]).astype(float)
Y, X = np.meshgrid(x, y)

pts = np.dstack((X,Y)).reshape((-1,2))
center = np.array([   x[int(len(x)/2.)], y[int(len(y)/2.)] ])#pts.mean(axis=0)
rot_pts = (pts-center).dot(rot(np.radians(0.)).T) + center
diffs = rot_pts - pts
u3 = diffs.reshape(u.shape)
# u = u3

rot_pts = rot_pts.reshape(u.shape)
X2, Y2 = rot_pts[:,:,0], rot_pts[:,:,1]
k = np.nonzero(init_phi == 1)
print k
print X[k], Y[k]

#u3 = np.empty_like(u)
#u3[:,:,0] = Y - (WORLD_MAX[1]-WORLD_MIN[1])/2.
#u3[:,:,1] = -(X - (WORLD_MAX[0]-WORLD_MIN[0])/2.)
#u3 *= .1/abs(u3).max()
#u = u3

#u2 = np.empty_like(u)
#u2[:,:,0] = u[:,:,1]
#u2[:,:,1] = u[:,:,0]
#u = u2

fig = plt.figure()

ax = fig.add_subplot(221, aspect='equal')
#mask = Y[:,:] == 15
#u[:,:,0][np.logical_not(mask)] = 0
#u[:,:,1][np.logical_not(mask)] = 0
print u
ax.quiver(X, Y, u[:,:,0], u[:,:,1], angles='xy', scale_units='xy', scale=1.)

init_phi_surf = sqp_problem.make_interp(init_phi)
plot_phi(fig, 222, init_phi_surf)

out_phi = apply_flow(init_phi, u)
apply_flow_forwards(fig, 224, init_phi, u, X2.ravel(), Y2.ravel(), init_phi.ravel())

#for i in range(20):
#  print i
#  out_phi = apply_flow(out_phi, u)

out_phi_surf = sqp_problem.make_interp(out_phi)
ax = plot_phi(fig, 223, out_phi_surf)
# ax.quiver(X, Y, np.fliplr(u[:,:,0].T), u[:,:,1].T, pivot='tip')

plt.show()

#out_phi = apply_flow(init_phi, u)
