import numpy as np
import skfmm
from skimage import measure as sm
from scipy import ndimage


def get_sdf(img):
  """
  given an image (m,n,3) it returns the signed distance field.
  """
  
  def flood_fill(mask):
    """
    All entries inside the boundary are +1, outside -1.
    Starts filling from (0,0) [assumes (0,0) is outside.
    
     - Mask is a matrix of 0s and 1s.
     - Boundary is specified with 1s. Background with 0s.
    """
    def in_range(x,y):
      nx,ny = mask.shape
      return (0 <= x < nx) and (0 <= y < ny) 

    seg       = np.ones_like(mask, dtype='float64')
    processed = np.zeros_like(mask, dtype='bool')
    queue     = [(0,0)]

    while len(queue) != 0:
      cx,cy = queue.pop()
      processed[cx,cy] = True 

      ## check if in the background:
      if mask[cx,cy]==0: 
        seg[cx,cy] = -1
        ## add neighbors:
        for dx,dy in [(-1,0), (1,0), (0,-1), (0,1)]:
          mx,my = cx+dx, cy+dy
          if in_range(mx,my) and not processed[mx,my]:        
            queue.append([mx,my])
    return seg

  mask = ((img[:,:,0] != 255) | (img[:,:,1] != 255) | (img[:,:,2] != 255)).astype('float64')
  seg_mask = flood_fill(mask)
  sdf = skfmm.distance(seg_mask)
  return sdf


  
def eval_zero_crossings(state_sdf, estimate_tsdf, plt=None):
  """
  return integrate_{x | tsdf(x) = 0} |state_sdf(x)| dx
  """
  z_cross   = sm.find_contours(estimate_tsdf, 0.0)

  if plt !=None:
    def m_plot(sdf, zc):
      plt.imshow(sdf.T, aspect=1, origin='lower')
      plt.xticks([])
      plt.yticks([])
      for contour in z_cross:    
        plt.plot(contour[:,0], contour[:,1], 'r')
      plt.colorbar()
      
    plt.subplot(1,2,1)
    m_plot(estimate_tsdf, z_cross)
    plt.title('estimate tsdf')
    plt.subplot(1,2,2)
    m_plot(state_sdf, z_cross)
    plt.title('state sdf')
    plt.show()

  def score_one_contour(contour):
    nsegs   = contour.shape[0]-1   
    mid_idx = np.arange(0.5, nsegs, 1.)
    mid_x   = np.interp(mid_idx, np.arange(nsegs+1), contour[:,0])
    mid_y   = np.interp(mid_idx, np.arange(nsegs+1), contour[:,1])
    mid_pt  = np.c_[mid_x, mid_y]
    seg_len = np.linalg.norm(contour[:-1,:] - contour[1:,:], axis=1)
    f_segs  = ndimage.map_coordinates(state_sdf, mid_pt.T, order=1)

    return np.sum(np.abs(f_segs)*seg_len), np.sum(seg_len)

  tot_score = 0.0
  tot_len   = 0.0
  print " num contours  : ", len(z_cross)
  for zc in z_cross:
    val, slen = score_one_contour(zc)
    tot_score += val
    tot_len   += slen
  return tot_score/tot_len


def test_zc_scoring():
  
  def make_square_img(SIZE):
    a = np.empty((SIZE, SIZE), dtype=np.uint8); a.fill(255)
    a[int(SIZE/4.),int(SIZE/4.):-int(SIZE/4.)] = 0
    a[int(SIZE/4.):-int(SIZE/4.),int(SIZE/4.)] = 0
    a[int(SIZE/4.):-int(SIZE/4.),-int(SIZE/4.)] = 0
    a[-int(SIZE/4.),int(SIZE/4.):-int(SIZE/4.)+1] = 0
    img = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
    for i in range(3):
      img[:,:,i] = a
    return img

  img1 = make_square_img(100)
  img2 = ndimage.interpolation.rotate(img1, 45, cval=255, order=1, reshape=False)
  sdf1 = get_sdf(img1) 
  sdf2 = get_sdf(img2)

  scores = {}
  import matplotlib.pylab as plt
  scores['11'] = eval_zero_crossings(sdf1, sdf1)
  scores['12'] = eval_zero_crossings(sdf1, sdf2)
  scores['21'] = eval_zero_crossings(sdf2, sdf1)
  scores['22'] = eval_zero_crossings(sdf2, sdf2)

  print scores 
  

if __name__=='__main__':
  test_zc_scoring()
  
  