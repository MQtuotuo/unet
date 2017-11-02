from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import os
from skimage.transform import resize
from skimage.io import imsave
from keras.layers.noise import GaussianNoise
import numpy as np
from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras import optimizers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard, ReduceLROnPlateau

from keras import backend as K
from pseudoRGB import pseudoRGB
import cv2

from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 256
img_cols = 256

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet(batchNormalization = False, dropout = None):
    inputs = Input((img_rows, img_cols, 3))
    
    encoded = GaussianNoise(0.05)(inputs)
    conv1 = Conv2D(16, (3, 3), padding="same", activation="relu")(encoded)
    if batchNormalization == True:
        conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, (3, 3), padding="same", activation="relu")(conv1)
    pool1 = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), padding="same", activation="relu")(pool1)
    if batchNormalization == True:
        conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv2)
    pool2 = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64,  (3, 3), padding="same", activation="relu")(pool2)
    if batchNormalization == True:
        conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64,  (3, 3), padding="same", activation="relu")(conv3)
    pool3 = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128,  (3, 3), padding="same", activation="relu")(pool3)
    if batchNormalization == True:
        conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128,  (3, 3), padding="same", activation="relu")(conv4)
    pool4 = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256,  (3, 3), padding="same", activation="relu")(pool4)
    conv5 = Conv2D(256,  (3, 3), padding="same", activation="relu")(conv5)

    up6 = UpSampling2D(size=(2,2), name="upsample1")(conv5)
    up6 = Concatenate(axis=3)([up6, conv4])
    conv6 = Conv2D(128,  (3, 3), padding="same", activation="relu")(up6)
    conv6 = Conv2D(128,  (3, 3), padding="same", activation="relu")(conv6)

    up7 = UpSampling2D(size=(2,2), name="upsample2")(conv6)
    up7 = Concatenate(axis=3)([up7, conv3])
    conv7 = Conv2D(64,  (3, 3), padding="same", activation="relu")(up7)
    conv7 = Conv2D(64,  (3, 3), padding="same", activation="relu")(conv7)

    up8 = UpSampling2D(size=(2,2), name="upsample3")(conv7)
    up8 = Concatenate(axis=3)([up8, conv2])
    conv8 = Conv2D(32,  (3, 3), padding="same", activation="relu")(up8)
    conv8 = Conv2D(32,  (3, 3), padding="same", activation="relu")(conv8)

    up9 = UpSampling2D(size=(2,2), name="upsample4")(conv8)
    up9 = Concatenate(axis=3)([up9, conv1])
    conv9 = Conv2D(16,  (3, 3), padding="same", activation="relu")(up9)
    if dropout is not None:
        conv9 = Dropout(dropout)(conv9)
    conv9 = Conv2D(16,  (3, 3), padding="same", activation="relu")(conv9)
    if dropout is not None:
        conv9 = Dropout(dropout)(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    
    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    
    return model

def get_UNetSimple (batchNormalization = False):
    inputs = Input((img_rows, img_cols, 3))
    encoded = GaussianNoise(0.05)(inputs)
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(encoded)
    if batchNormalization == True:
        conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv1)
    pool1 = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(pool1)
    if batchNormalization == True:
        conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv2)
    pool2 = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128,  (3, 3), padding="same", activation="relu")(pool2)
    if batchNormalization == True:
        conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128,  (3, 3), padding="same", activation="relu")(conv3)
    pool3 = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256,  (3, 3), padding="same", activation="relu")(pool3)
    if batchNormalization == True:
        conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256,  (3, 3), padding="same", activation="relu")(conv4)
    pool4 = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(conv4)



    conv4a = Conv2D(512,  (3, 3), padding="same", activation="relu")(pool4)
    if batchNormalization == True:
        conv4a = BatchNormalization()(conv4a)
    conv4a = Conv2D(512,  (3, 3), padding="same", activation="relu")(conv4a)
    pool4a = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(conv4a)

    conv4b = Conv2D(512,  (3, 3), padding="same", activation="relu")(pool4a)
    if batchNormalization == True:
        conv4b = BatchNormalization()(conv4b)
    conv4b = Conv2D(512,  (3, 3), padding="same", activation="relu")(conv4b)
    pool4b = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(conv4b)


    conv5 = Conv2D(1024,  (3, 3), padding="same", activation="relu")(pool4b)
    conv5 = Conv2D(1024,  (3, 3), padding="same", activation="relu")(conv5)


    up6b = UpSampling2D(size=(2,2), name="upsample1b")(conv5)
    up6b = Concatenate(axis=3)([up6b, conv4b])
    conv6b = Conv2D(512,  (3, 3), padding="same", activation="relu")(up6b)
    conv6b = Conv2D(512,  (3, 3), padding="same", activation="relu")(conv6b)


    up6a = UpSampling2D(size=(2,2), name="upsample1a")(conv6b)
    up6a = Concatenate(axis=3)([up6a, conv4a])
    conv6a = Conv2D(512,  (3, 3), padding="same", activation="relu")(up6a)
    conv6a = Conv2D(512,  (3, 3), padding="same", activation="relu")(conv6a)


    up6 = UpSampling2D(size=(2,2), name="upsample1")(conv6a)
    up6 = Concatenate(axis=3)([up6, conv4])
    conv6 = Conv2D(256,  (3, 3), padding="same", activation="relu")(up6)
    conv6 = Conv2D(256,  (3, 3), padding="same", activation="relu")(conv6)

    up7 = UpSampling2D(size=(2,2), name="upsample2")(conv6)
    up7 = Concatenate(axis=3)([up7, conv3])
    conv7 = Conv2D(128,  (3, 3), padding="same", activation="relu")(up7)
    conv7 = Conv2D(128,  (3, 3), padding="same", activation="relu")(conv7)

    up8 = UpSampling2D(size=(2,2), name="upsample3")(conv7)
    up8 = Concatenate(axis=3)([up8, conv2])
    conv8 = Conv2D(64,  (3, 3), padding="same", activation="relu")(up8)
    conv8 = Conv2D(64,  (3, 3), padding="same", activation="relu")(conv8)

    up9 = UpSampling2D(size=(2,2), name="upsample4")(conv8)
    up9 = Concatenate(axis=3)([up9, conv1])
    conv9 = Conv2D(32,  (3, 3), padding="same", activation="relu")(up9)
    conv9 = Conv2D(32,  (3, 3), padding="same", activation="relu")(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def preprocess_image(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 3), dtype=np.uint8)
   
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)
   

        #imgs_temp = resize(imgs[i], (img_rows, img_cols), preserve_range=True)
        #imgs_temp = cv2.cvtColor(imgs_temp, cv2.COLOR_BGR2GRAY)
        #print(imgs_temp.shape)
        #imgs_temp = pseudoRGB(imgs_temp, "clahe", visualize=False)
        
        #print(imgs_temp.shape)
        #imgs_p[i] = imgs_temp

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def preprocess_mask(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 1), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    #imgs_train = preprocess_image(imgs_train)
    #imgs_mask_train = preprocess_mask(imgs_mask_train)

    #imgs_train = imgs_train.astype('float32')
    #mean = np.mean(imgs_train)  # mean for data centering
    #std = np.std(imgs_train)  # std for data normalization

    #imgs_train -= mean
    #imgs_train /= std
    #imgs_train = np.squeeze(imgs_train)
    datagen = ImageDataGenerator(rotation_range=30, height_shift_range=0.01, horizontal_flip=True)
    # fit parameters from data
    datagen.fit(imgs_train)
    print(imgs_train.shape)

    #imgs_mask_train = imgs_mask_train.astype('float32')
    #imgs_mask_train /= 255.  # scale masks to [0, 1]
    print(imgs_mask_train.shape)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    #model = get_unet(batchNormalization = True, dropout = 0.2)
    model = get_UNetSimple(batchNormalization = True)
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    stopping = EarlyStopping (monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    
    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=200, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint, stopping, reduceLR])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    #imgs_test = preprocess_image(imgs_test)

    #imgs_test = imgs_test.astype('float32')
    #mean_test = np.mean(imgs_test)  # mean for data centering
    #std_test = np.std(imgs_test)  # std for data normalization
   
    #imgs_test -= mean_test
    #imgs_test /= std_test
    
    #imgs_test = np.squeeze(imgs_test)
    print(imgs_test.shape)

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

if __name__ == '__main__':
    train_and_predict()
