from __future__ import division
import numpy as np
import cv2

def clip_line(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
    if not np.allclose(x0, x1):
        slope = (y1 - y0) / (x1 - x0)
        y_intercept = y0 - slope*x0
        if xmin is not None:
            if x0 < xmin: x0, y0 = xmin, slope*xmin + y_intercept
            if x1 < xmin: x1, y1 = xmin, slope*xmin + y_intercept
        if xmax is not None:
            if x0 > xmax: x0, y0 = xmax, slope*xmax + y_intercept
            if x1 > xmax: x1, y1 = xmax, slope*xmax + y_intercept

    if not np.allclose(y0, y1):
        inv_slope = (x1 - x0) / (y1 - y0)
        x_intercept = x0 - inv_slope*y0
        if ymin is not None:
            if y0 < ymin: x0, y0 = inv_slope*ymin + x_intercept, ymin
            if y1 < ymin: x1, y1 = inv_slope*ymin + x_intercept, ymin
        if ymax is not None:
            if y0 > ymax: x0, y0 = inv_slope*ymax + x_intercept, ymax
            if y1 > ymax: x1, y1 = inv_slope*ymax + x_intercept, ymax

    return x0, y0, x1, y1

def project(f, cx, X, Y):
    return f*X/Y + cx, Y

def unproject(f, cx, x, y):
    return (x - cx)/f*y, y

class Renderer(object):
    def __init__(self, P):
        """
        P = camera's projection matrix
        """
        self.P = P
    def line(self, XY0, XY1):
        abstract
    def drawpoly(self, XYs, complete_poly=False):
        it = iter(XYs)
        assert len(XYs) >= 2
        firstXY = XY0 = it.next()
        while True:
            try:
                XY1 = it.next()
                self.line(XY0, XY1)
                XY0 = XY1
            except StopIteration:
                break
        if complete_poly:
            self.line(XY1, firstXY)

    def fillpoly(self, XYs):
        abstract

    def point(self, XY):
        abstract

class Render1d(Renderer):
    
    def __init__(self, t, r_angle, fov, width):
        
        """P is 3 X 2: homogeneous 2d -> homogeneous 1d"""

        self.f = width / (2*np.tan(fov/2.))
        self.cx = width/2.-.5

        Rt = np.empty((2,3))
        Rt[:2,:2] = rotation2d(r_angle)
        Rt[0,2] = t[0]
        Rt[1,2] = t[1]
        self.Rt = np.linalg.inv(np.r_[Rt, [[0,0,1]]])[:2,:3]
        
        self.width = width
        self.image = np.empty((width,3))
        self.image.fill(0)
        self.depth = np.empty(width)
        self.depth.fill(np.inf)

        self.color = (1.,1.,1.)
        
    def line(self, XY0, XY1):
        X0c, Y0c = self.Rt[:2,:2].dot(XY0) + self.Rt[:2,2]
        X1c, Y1c = self.Rt[:2,:2].dot(XY1) + self.Rt[:2,2]

        # clip line to near plane
        clip_near = .001
        if Y0c < clip_near and Y1c < clip_near:
            return
        X0c, Y0c, X1c, Y1c = clip_line(X0c, Y0c, X1c, Y1c, xmin=None, xmax=None, ymin=clip_near, ymax=None)

        # project line onto screen
        x0 = project(self.f, self.cx, X0c, Y0c)[0]
        x1 = project(self.f, self.cx, X1c, Y1c)[0]
        if (x0 < 0 and x1 < 0) or (x0 > self.width-1 and x1 > self.width-1):
            return
        if x1 < x0:
            x0, x1 = x1, x0
            Y0c, Y1c = Y1c, Y0c
            X0c, X1c = X1c, X0c
        x0int, x1int = int(round(x0)), int(round(x1))
        npix = x1int-x0int+1
        xints = np.arange(x0int, x1int+1)
        new_depth_slice = np.interp(xints, *project(self.f, self.cx, np.linspace(X0c, X1c, npix), np.linspace(Y0c, Y1c, npix)))
        # clip projected line to screen
        in_frustum = (xints >= 0) & (xints <= self.width-1)
        xints = xints[in_frustum]
        assert np.alltrue(xints == np.arange(xints[0], xints[-1]+1)) # xints should be contiguous
        new_depth_slice = new_depth_slice[in_frustum]

        # draw onto screen, respecting depth
        im_slice = self.image[xints[0]:xints[-1]+1]
        depth_slice = self.depth[xints[0]:xints[-1]+1]
        in_front = new_depth_slice < depth_slice
        if np.any(in_front):
            im_slice[in_front] = self.color
            depth_slice[in_front] = new_depth_slice[in_front]

    def fillpoly(self, XYs):
        self.drawpoly(XYs, complete_poly=True)
        

    
class Render2d(Renderer):
    def __init__(self, XY_bottomleft, XY_topright, width):
        Xl, Yb = XY_bottomleft
        Xr, Yt = XY_topright
        height = int(round( width * (Yt - Yb) / (Xr - Xl) ))
        
        scaling = width / (Xl - Xr) 
        self.P = np.zeros((3,3))
        self.P[0,0] = scaling
        self.P[1,1] = scaling
        self.P[2,2] = 1
        self.P[0,2] = width/2. - .5
        self.P[1,2] = height/2. - .5
        
        self.width = width
        self.height = height
        self.image = np.empty((width, height,3))
        self.image.fill(0)
        self.color = (1.,1.,1.)
    def line(self, XY0, XY1):
        x0, y0 = self.P[:2,:2].dot(XY0) + self.P[:2,2]
        x1, y1 = self.P[:2,:2].dot(XY1) + self.P[:2,2]
        cv2.line(self.image, (int(x0), int(y0)), (int(x1), int(y1)), self.color)
    def fillpoly(self, XYs):
        XYs = np.asarray(XYs)
        xys = XYs.dot(self.P[:2,:2].T) + self.P[:2,2][None,:]
        cv2.fillPoly(self.image, [np.int32(xys)], self.color)
    def point(self, XY):
        x, y = self.P[:2,:2].dot(XY) + self.P[:2,2]
        if 0 <= int(y) < self.image.shape[0] and 0 <= int(x) < self.image.shape[1]:
            self.image[int(y), int(x), :] = self.color
        #cv2.circle(self.image, (int(x), int(y)), 1, self.color)

class Drawable(object):
    def draw(self, renderer):
        abstract

class Polygon(Drawable):
    def __init__(self, vertices, color=(1,1,1), filled=False):
        self.vertices = vertices
        self.color = color
        self.filled = filled
    def draw(self, renderer):
        renderer.color = self.color
        if self.filled:
            renderer.fillpoly(self.vertices)
        else:
            renderer.drawpoly(self.vertices, complete_poly=True)

class Point(Drawable):
    def __init__(self, pt, color=(1,1,1)):
        self.pt = pt
        self.color = color
    def draw(self, renderer):
        renderer.color = self.color
        renderer.point(self.pt)

def make_camera_poly(t, r_angle, fov):
    t = np.array(t)
    R = rotation2d(r_angle)
    pt1 = t + R.dot([np.sin(fov/2.), np.cos(fov/2.)])
    pt2 = t + R.dot([-np.sin(fov/2.), np.cos(fov/2.)])
    return Polygon([t,pt1, pt2])
    
def rotation2d(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def show_2d_image(image, windowname="default2d"):
    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(windowname, image)
def show_1d_image(images, windowname="default1d"):
    images = np.atleast_3d(np.asarray(images))
    thick_images = []
    for i in range(images.shape[0]):
        thick_images.append(np.tile(images[i][None,:], (20, 1, 1)))
    image2d = np.concatenate(thick_images)
    cv2.imshow(windowname, image2d)

class Camera1d(object):
    def __init__(self, t, r_angle, fov, width):
        self.t = np.asarray(t)
        self.r_angle = r_angle
        self.fov = fov
        self.width = width

    def render(self, drawables):
        renderer = Render1d(self.t, self.r_angle, self.fov, self.width)
        for drawable in drawables:
            drawable.draw(renderer)
        return renderer.image, renderer.depth

    def unproject(self, depth):
        f = self.width / (2*np.tan(self.fov/2.))
        cx = self.width/2.-.5
        R = rotation2d(self.r_angle)

        x = np.arange(0,self.width)
        Xc, Yc = unproject(f, cx, x, depth)

        return np.c_[Xc,Yc].dot(R.T) + self.t[None,:]


class Camera2d(object):
    def __init__(self, XY_bottomleft, XY_topright, width):
        self.bl = XY_bottomleft
        self.tr = XY_topright
        self.width = width
    def render(self, drawables):
        renderer = Render2d(self.bl, self.tr, self.width)
        for drawable in drawables:
            drawable.draw(renderer)
        return renderer.image

def draw_points(P, XYs, image, color=(1.,0,1.)):
    Xints, Yints = np.int32(np.round(XYs)).T
    goodinds = (Xints >= 0) & (Yints >= 0) & (Xints < image.shape[1]) & (Yints < image.shape[0])
    image[Yints[goodinds], Xints[goodinds]] = color

def main():

    poly = Polygon([[.2, .2], [0,1], [1,1], [1,.5]])
    # poly = Polygon([[0,.8], [1,.5]])
    t = (0, -.5)
    r_angle = 0
    fov = 75 * np.pi/180.

    cam1d = Camera1d(t, r_angle, fov, 500)
    cam2d = Camera2d((-1,-1), (1,1), 500)
    
    while True:
        image1d,depth1d = cam1d.render([poly])

        depth_min, depth_max = 0, 1
        depth1d_normalized = (np.clip(depth1d, depth_min, depth_max) - depth_min)/(depth_max - depth_min)
        depth1d_image = np.array([[.5, 0, 0]])*depth1d_normalized[:,None] + np.array([[1., 1., 1.]])*(1. - depth1d_normalized[:,None])
        depth1d_image[np.logical_not(np.isfinite(depth1d))] = (0, 0, 0)

        observed_XYs = cam1d.unproject(depth1d)

        renderlist = [poly, make_camera_poly(t, cam1d.r_angle, fov)] + [Point(p, c) for (p, c) in zip(observed_XYs, depth1d_image) if np.isfinite(p).all()]
        image2d = cam2d.render(renderlist)

        show_1d_image([image1d, depth1d_image], "image1d+depth1d")
        show_2d_image(image2d)
        key = cv2.waitKey() & 255
        print "key", key

        # mac
        # if key == 63234:
        #     cam1d.r_angle -= .1
        # elif key == 63235:
        #     cam1d.r_angle += .1
        # elif key == 113:
        #     break

        # linux
        if key == 81:
            cam1d.r_angle -= .1
        elif key == 83:
            cam1d.r_angle += .1
        elif key == ord('q'):
            break
    
if __name__ == "__main__":
    main()
