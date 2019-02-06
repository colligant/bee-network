import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
import tensorflow as tf
from glob import glob
from data_utils_polygons import generate_class_mask 
import numpy as np

NO_DATA = 999


def custom_objective(y_true, y_pred):
    '''I want to mask all values that 
       are not data, given a y_true 
       that is masked. '''
    masked = tf.equal(y_true, NO_DATA)
    y_true_mask = tf.boolean_mask(y_true, masked)
    y_pred_mask = tf.boolean_mask(y_pred, masked)
    return tf.keras.losses.binary_crossentropy(y_true_mask, y_pred_mask)

def fcnn_model(image_shape, n_classes):

    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
        input_shape=image_shape, data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=n_classes, kernel_size=1, padding='same',
        activation='softmax')) # 1x1 convolutions for pixel-wise prediciton.
    model.add(tf.keras.layers.Flatten())
    # Take a look at the model summary
    model.summary()
    return model

def create_model(image_shape, n_classes):
    model = fcnn_model(image_shape, n_classes)
    model.compile(loss=custom_objective,
                 optimizer='adam',
                 metrics=['accuracy'])
    return model

def one_hot_encoding(class_mask, n_classes):
    '''Assumes classes range from 0 -> (n-1)'''
    shp = class_mask.shape
    out = np.ones((shp[0], shp[1], n_classes))*NO_DATA
    for i in range(n_classes):
        out[:, :, i][class_mask == i] = i
    return out


def train_model(image_directory, image_shape):

    n_classes = 2
    model = create_model(image_shape, n_classes)
    


    for f in glob(image_directory + "*.json"):
        jpg = f[:-13] + ".jpg"
        class_mask, input_image = generate_class_mask(f, jpg)
        class_mask[class_mask == -1] = NO_DATA 
        class_mask = class_mask.astype(np.int32)
        if not len(class_mask[class_mask == 1]):
            # Have to check if bees are present in the
            # image. They should bee (haha).
            continue
        class_mask = one_hot_encoding(class_mask, n_classes) 

        in_img = np.zeros((1, 1080, 1920, 3))
        in_class = np.zeros((1, 1080, 1920, n_classes))

        in_img[0, :, :, :] = input_image
        in_class[0, :, :, :] = class_mask

        model.fit(in_img, 
                 in_class,
                 epochs=2)

if __name__ == '__main__':
    path = '/home/thomas/bee-network/for_bees/Blank VS Scented/B VS S Day 1/Frames JPG/'
    shape = (1080, 1920, 3)
    train_model(path, shape)
