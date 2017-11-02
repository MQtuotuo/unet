import cv2
import numpy as np

def pseudoRGB(img, method="clahe", visualize=False):
    if method not in ["clahe"]:
        exit("Pseudo RGB method " + str(method) + " is unknown.")

    conversionFactor = 256
    if img.dtype == np.uint8:
        conversionFactor = 1
        method = 'clahe'

    if method == "clahe":
        if img.shape[0] > 512 or img.shape[1] > 512:
            factor = max(img.shape[0], img.shape[1]) / 512
            clipfactor = 256
        else:
            factor = 1
            clipfactor = 1

        clahe = cv2.createCLAHE(clipLimit=8.0 * clipfactor, tileGridSize=(int(2 * factor), int(2 * factor)))
        red = (clahe.apply(img) / conversionFactor).astype('uint8')
        clahe = cv2.createCLAHE(clipLimit=2.0 * clipfactor, tileGridSize=(int(8 * factor), int(8 * factor)))
        blue = (clahe.apply(img) / conversionFactor).astype('uint8')
        clahe = cv2.createCLAHE(clipLimit=4.0 * clipfactor, tileGridSize=(int(4 * factor), int(4 * factor)))
        green = (clahe.apply(img) / conversionFactor).astype('uint8')

        img = cv2.merge((blue, green, red))

        if visualize == True:
            cv2.imshow('image512', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            for i in range(1, 5):
                cv2.waitKey(1)

    return img
