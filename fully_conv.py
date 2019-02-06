import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras.backend as K
import tensorflow as tf
tf.enable_eager_execution()
from glob import glob
import matplotlib.pyplot as plt
import time
from data_utils_polygons import generate_class_mask 
import numpy as np

NO_DATA = 3

def custom_objective(y_true, y_pred):
    '''I want to mask all values that 
       are not data, given a y_true 
       that has NODATA values. '''
    y_true = tf.reshape(y_true, (1080*1920, 2))
    y_pred = tf.reshape(y_pred, (1080*1920, 2))
    masked = tf.not_equal(y_true, NO_DATA)
    y_true_mask = tf.boolean_mask(y_true, masked)
    y_pred_mask = tf.boolean_mask(y_pred, masked)
    return tf.keras.losses.binary_crossentropy(y_true_mask, y_pred_mask)

def fcnn_model(image_shape, n_classes):

    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
        input_shape=image_shape, data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
    #model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'))
    #model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=n_classes, kernel_size=1, padding='same',
        activation='softmax')) # 1x1 convolutions for pixel-wise prediciton.
    # Take a look at the model summary
    model.summary()
    return model

def create_model(image_shape, n_classes):
    model = fcnn_model(image_shape, n_classes)
    model.compile(loss=custom_objective,
                 optimizer=tf.train.AdamOptimizer(),
                 metrics=['accuracy'])
    return model

def one_hot_encoding(class_mask, n_classes):
    '''Assumes classes range from 0 -> (n-1)'''
    shp = class_mask.shape
    out = np.ones((shp[0], shp[1], n_classes))*NO_DATA
    for i in range(n_classes):
        out[:, :, i][class_mask == i] = 1
    return out

def train_model(image_directory, image_shape):

    n_classes = 2
    model = create_model(image_shape, n_classes)
    
    fname = None 
    jj = None
    for i in range(3):
        for f in glob(image_directory + "*.json"):
            jpg = f[:-13] + ".jpg"
            class_mask, input_image = generate_class_mask(f, jpg)
            if class_mask is None:
                continue
            input_image = input_image.astype(np.float32)
            class_mask[class_mask == -1] = NO_DATA 
            class_mask = class_mask.astype(np.int32)

            fname = f
            jj = jpg

            class_mask = one_hot_encoding(class_mask, n_classes) 
            in_class = np.zeros((1, 1080, 1920, n_classes))
            in_img = np.zeros((1, 1080, 1920, 3))
            in_class[0, :, :, :] = class_mask
            in_img[0, :, :, :] = input_image

            model.fit(in_img, 
                     in_class,
                     epochs=1)
     
    mask, im = generate_class_mask(fname, jj)
    in_img = np.zeros((1, 1080, 1920, 3))
    in_img[0, :, :, :] = im

    start = time.time()
    out = model.predict(in_img)
    end = time.time()
    print("T_pred:", end-start)
    out1 = out[0:, :, :, 0]
    out2 = out[0:, :, :, 1]
    out1 = np.reshape(out1, (1080, 1920))
    out2 = np.reshape(out2, (1080, 1920))
    fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
    ax[0].imshow(out1)
    ax[1].imshow(out2)
    ax[1].set_title("Bee prediction")
    plt.show()

if __name__ == '__main__':
    path = '/home/thomas/bee-network/for_bees/Blank VS Scented/B VS S Day 1/Frames JPG/'
    shape = (1080, 1920, 3)
    train_model(path, shape)
