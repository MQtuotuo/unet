from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


# resize. stupid, isnt there a python routine for that?..
def resizeToFit (img, imageShape, verbose = False):
    if len(imageShape) < 3:
        img = img.reshape(img.shape[0], img.shape[1], 1)
        nChannels = 1
    else:
        nChannels = imageShape[2]
    newImageShape = (imageShape[0], imageShape[1], nChannels)
    canvas = np.zeros(newImageShape, img.dtype)
    
    #print ("img shape: " + str(img.shape))
    #print ("newImageShape shape: " + str(newImageShape))

    # resize to fit
    if img.shape[0] > img.shape[1]:
        
        if verbose == True:
            print ("resizeToFit: More rows than cols")

        factor = imageShape[0]/img.shape[0]
        newImgShape = (int(img.shape[1]*factor), imageShape[0])
        tmpimg = cv2.resize(img, newImgShape , interpolation = cv2.INTER_CUBIC)
        tmpimg  = tmpimg.reshape(tmpimg.shape[0], tmpimg.shape[1], nChannels)
        pad = int((imageShape[1] - tmpimg.shape[1])/2)
        canvas[:, pad:pad+tmpimg.shape[1],:] = tmpimg
    else:
        if verbose == True:
            print ("resizeToFit: More cols than rows")
        factor = imageShape[1]/img.shape[1]
        newImgShape = (int(imageShape[1]), int(img.shape[0]*factor))
        #newImgShape = (imageShape[0], int(img.shape[1]*factor))

        if verbose == True:
            print ("resizeToFit: Resizing from " + str(imageShape) + " to " + str(newImgShape))

        # there is a damn bug in resize.  my input is 1711, 1711 (or so), desigred output (512,512),
        # but the output is (512,511)(!!!) -- avoid this by copying to :tmpimg.shape[1] below
        tmpimg = cv2.resize(img, newImgShape , interpolation = cv2.INTER_CUBIC)
        tmpimg  = tmpimg.reshape(tmpimg.shape[0], tmpimg.shape[1], nChannels)
        pad = int((imageShape[0] - tmpimg.shape[0])/2)
        canvas[pad:pad+tmpimg.shape[0],:tmpimg.shape[1],:] = tmpimg
    return canvas

