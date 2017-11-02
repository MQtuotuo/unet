from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

from skimage import measure
from skimage import morphology
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk



# clear noise of the predicted mask and get only the biggest mask
def getOneMask (mask, visualize = False):   
#    print(mask.shape)
    mask = mask[:,:,0]
#    print(mask.shape)

    labels = measure.label(mask)
    label_flat = labels.flatten()
    
    counts = np.bincount(label_flat)
#    print (counts)
    if len(counts) == 0:
        print ("WARNING: IMAGE IS BLACK! WILL RETURN WHOLE IMAGE!")
        new_mask = np.ones_like(mask)
        return new_mask 
    
    big_label_nr = np.argmax(counts[1:]) + 1
    new_mask = np.zeros_like(mask)
    new_mask = new_mask + np.where(labels==big_label_nr,1,0)
    structureElement = disk(5)
    new_mask = morphology.dilation(new_mask, structureElement)

    return new_mask

