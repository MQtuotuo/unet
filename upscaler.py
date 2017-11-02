from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


# upscale a small mask given the original image
def upscaler (smallMask, imgShape, verbose = True):
    # the possibilities: img is color or smallMask is color or both are or are not.
    # if both the same, all good.
    # if not, the color space of the image does not matter.
    # but the color space of the mask does matter, as we upscale to the same color space.
    
    # just check, if we were given an image
    if hasattr(imgShape, "shape"):
        imgShape = img.shape
        if verbose: 
            print ("Originally image has shape:" + str(imgShape))
    
    if len(smallMask.shape) < 3:
        smallMask = smallMask.reshape(smallMask.shape[0], smallMask.shape[1], 1)
        nChannels = 1
    else:
        nChannels = smallMask.shape[2] 

    if verbose: 
        print ("Image has shape:" + str(imgShape[:2]) + "," + str(nChannels))

    # resize to fit
    if imgShape[0] > imgShape[1]:
        if verbose:
            print ("More rows than cols.")
            
        # determine original shrinking factor
        factor = smallMask.shape[0]/imgShape[0]
        if verbose: 
            print ("Factor: " + str(factor))
        
        # determine shape of shrinked image
        smallShape = (smallMask.shape[0], int(imgShape[1]*factor))
        if verbose: 
            print ("Small shape:" + str (smallShape))
        
        # determine the pad that was applied before
        pad = int((smallMask.shape[1] - smallShape [1])/2)
        if verbose: 
            print ("Pad: " + str(pad))

        # remove the pad
        smallMask = smallMask[:, pad:smallMask.shape[1]-pad ].copy()
        if verbose: 
            print (smallMask.shape)

        # copy and upscale to original image size
        upscaledSize = (imgShape[1], imgShape[0]) # thanks, opencv!
        if verbose: 
            print (upscaledSize)

        upscaled = cv2.resize(smallMask, upscaledSize, interpolation = cv2.INTER_CUBIC)
    else:
        if verbose:
            print ("More cols than rows.")
            
        # determine original shrinking factor
        factor = smallMask.shape[1]/imgShape[1]
        if verbose: 
            print ("Factor: " + str(factor))
        
        # determine shape of shrinked image
        smallShape = (int(imgShape[0]*factor), smallMask.shape[1])
        if verbose: 
            print ("Small shape:" + str (smallShape))
        
        # determine the pad that was applied before
        pad = int((smallMask.shape[0] - smallShape [0])/2)
        if verbose: 
            print ("Pad: " + str(pad))

        # remove the pad
        smallMask = smallMask[pad:smallMask.shape[0]-pad, :].copy()
        if verbose: 
            print (smallMask.shape)

        # copy and upscale to original image size
        upscaledSize = (imgShape[1], imgShape[0]) # thanks, opencv!
        if verbose: 
            print (upscaledSize)
        
        upscaled = cv2.resize(smallMask, upscaledSize, interpolation = cv2.INTER_CUBIC)
    upscaled = upscaled.reshape(upscaled.shape[0], upscaled.shape[1], nChannels)
    return upscaled 

