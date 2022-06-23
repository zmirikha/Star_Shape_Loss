from __future__ import print_function
import os
import numpy as np
from PIL import Image


def unit_vector(vector):
    
    """ Returns the unit vector of the vector.  """
    
    return vector / (np.linalg.norm(vector)+(1e-10))

def cos_between(v1, v2):
    
    """ Calculate the cosine of the angle between two vectors. """
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)

def neigh_map(img):
    
    """
    Find target object center and quantize the possible angles of all lines
    passing through the center to d directions and generate a mask (called
    neighboring map/ regional map) to split the image domain into d regions.
    """
    
    pos = np.where(img>0)
    center = np.floor(np.mean(pos, axis=1))

    
    size_x = img.shape[0]
    size_y = img.shape[1]
    
    v_x = np.zeros((size_x,size_y))
    v_y = np.zeros((size_x,size_y))
    ng = np.zeros((size_x,size_y))
    
    for i in range(size_x):
     for j in range(size_y):

       v_x[i,j] = center[1]-i
       v_y[i,j] = center[0]-j

       if (v_x[i,j] >= 0 and v_y[i,j] >= 0):

          c = cos_between((v_x[i,j],v_y[i,j]), [1,0])

          if (c >= 0.9063 and c <= 1):
            ng[j,i] = 2
            
          elif (c >= 0 and c <= 0.2588):
            ng[j,i] = 4
            
          else:
            ng[j,i] = 3

       elif (v_x[i,j] > 0 and v_y[i,j] < 0):
      
          c = cos_between((v_x[i,j],v_y[i,j]), [1,0])

          if (c >= 0.9063 and c <= 1):
            ng[j,i] = 2
            
          elif (c >= 0 and c <= 0.2588):
            ng[j,i] = 8
            
          else:
            ng[j,i] = 1

       elif (v_x[i,j] < 0 and v_y[i,j] > 0):

          c = cos_between((v_x[i,j],v_y[i,j]), [1,0])

          if (c <= -0.9063 and c >= -1):
            ng[j,i] = 6
            
          elif (c <= 0 and c >= -0.2588):
            ng[j,i] = 4
            
          else:
            ng[j,i] = 5

       elif (v_x[i,j] <= 0 and v_y[i,j] <= 0):

          c = cos_between((v_x[i,j],v_y[i,j]), [1,0])

          if (c <= -0.9063 and c >= -1):
            ng[j,i] = 6
            
          elif (c <= 0 and c >= -0.2588):
            ng[j,i] = 8
            
          else:
            ng[j,i] = 7

    return ng



def main():
    
    gt_dir = input("enter gt directory:")
    ng_dir = input("enter neighboring maps directory:")

    images = os.listdir(gt_dir)

    for image in images:

        gt=Image.open(os.path.join(gt_dir, image))
        img=np.asarray(gt)
    
        # generate neighbouring maps from target ground truth maps
        ng=neigh_map(img)

        ng_img = Image.fromarray(np.uint8(ng))
        ng_img.save(os.path.join(ng_dir, image))



if __name__ == "__main__":
    main()


