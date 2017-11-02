from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import os
from skimage.transform import resize
from skimage.io import imsave
from keras.layers.noise import GaussianNoise
import numpy as np
from keras import backend as K
from pseudoRGB import pseudoRGB
from resizeToFit import *
from getOneMask import *
from upscaler import *
from skimage.io import imsave, imread
import cv2
from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
RSNA_path = '/Users/Ming/Documents/Bonn/MasterThesis/Bone/RSNA'
mask_path = 'preds/'
output_path = 'output/'


smooth = 1.


def preprocessing (x, y = None, resizeTo = None):
        # resize to intermediate size
        imageSize = (256, 256)
        maskSize = (256, 256)
        if resizeTo is None:
            resizeTo = imageSize
        x = resizeToFit (x, resizeTo)
        x = pseudoRGB (x, "clahe")
        
        if y is not None:
            if resizeTo is None:
                resizeTo = maskSize
            y = resizeToFit (y, resizeTo)
            y = y.astype('float32')/255
            return x, y
        return x


if not os.path.exists(output_path):
        os.mkdir(output_path)
print('-' * 30)
print('Loading and test data...')
print('-' * 30)

test_data_path = os.path.join(RSNA_path, 'train')
images = os.listdir(test_data_path)
total = len(images)
imgs_id = np.ndarray((total,), dtype=np.int32)


i = 0
print('-' * 30)
print('Loading test images...')
print('-' * 30)
for image_name in images:
    img_id = int(image_name.split('.')[0])
    imgs_id[i] = img_id
   

    if 'mask' in image_name:
        continue
    image_mask_name = image_name.split('.')[0] + '_pred.png'
    img = imread(os.path.join(test_data_path, image_name), as_grey=True)
    
    #print(img.shape)
    #print("##")
    img_mask = imread(os.path.join(mask_path, image_mask_name), as_grey=True)
    #print(img_mask.shape)

    # get the original image, preprocess it
    originalImage = img.copy()
    originalImageShape = originalImage.shape
    originalImage = preprocessing(img, resizeTo=originalImageShape)
    

    # upscale the mask
    predicted_mask = upscaler(img_mask, originalImageShape)
    predicted_mask = predicted_mask / np.max(predicted_mask)  
    #predicted_mask = predicted_mask * 255
    #print(predicted_mask)
    #print(predicted_mask.shape)
    #predicted_mask = getOneMask(predicted_mask)
    predicted_mask = np.squeeze(predicted_mask)

    #print(predicted_mask.shape)
    #print("-----")
    #print(originalImage.shape)
    #print(img.shape)
    for c in range(originalImage.shape[2]):
        originalImage[:, :, c] = originalImage[:, :, c] * predicted_mask
    imsave(os.path.join(output_path, str(img_id) + '_new.png'), originalImage)

print('Loading done.')

