import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras.backend as K
import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt
import time
from data_utils_polygons import generate_class_mask, normalize_image
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
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=8, padding='same', activation='relu',
        input_shape=image_shape, data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=4, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(filters=n_classes, kernel_size=2, padding='same',
        activation='softmax')) # 1x1 convolutions for pixel-wise prediciton.
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
        out[:, :, i][class_mask == i] = 1
    return out

def train_model(image_directory, image_shape, box_size, epochs=3):
    n_classes = 2
    model = create_model(image_shape, n_classes)
    for i in range(epochs):
        print("Epoch:", i)
        for f in glob(image_directory + "*.json"):
            jpg = f[:-13] + ".jpg"
            class_mask, input_image = generate_class_mask(f, jpg, box_size=bs)
            if class_mask is None:
                continue
            input_image = input_image.astype(np.float32)
            class_mask[class_mask == -1] = NO_DATA 
            class_mask = class_mask.astype(np.int32)
            class_mask = one_hot_encoding(class_mask, n_classes) 
            in_class = np.zeros((1, 1080, 1920, n_classes))
            in_img = np.zeros((1, 1080, 1920, 3))
            in_class[0, :, :, :] = class_mask
            in_img[0, :, :, :] = input_image
            model.fit(in_img, 
                     in_class,
                     epochs=1, verbose=1)
    return model

def evaluate_image(image_path, th=0.1):
    im = cv2.imread(image_path)
    im = normalize_image(im)
    in_img = np.zeros((1, 1080, 1920, 3))
    in_img[0, :, :, :] = im
    start = time.time()
    out = model.predict(in_img)
    end = time.time()
    print("T_pred:", end-start)
    out = out[0:, :, :, :]
    out = np.argmax(out, axis=3)
    out = np.reshape(out, (1080, 1920))
    return out

if __name__ == '__main__':
    path = '/home/thomas/bee-network/for_bees/Blank VS Scented/B VS S Day 1/Frames JPG/'
    shape = (1080, 1920, 3)

    for bs in range(20, 33, 3):
        model_path = 'models/fcnn_bs{}.h5'.format(bs)
        if not os.path.isfile(model_path): 
            model = train_model(path, shape, bs)
            model.save(model_path)
        else:
            model = tf.keras.models.load_model(model_path,
                    custom_objects={'custom_objective':custom_objective})

        gt = 'ground_truth.jpg'
        json = 'ground_truth_labels.json'
        class_mask, in_img = generate_class_mask(json, gt)
        class_mask[class_mask == -1] = 0
        predictions = evaluate_image(gt)
        num = len(class_mask[class_mask == predictions])
        acc = num / (1080*1920)
        fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
        ax[0].imshow(predictions)
        ax[0].set_title("Preds, box_size={}, acc={:.3f}".format(bs, acc))
        ax[1].imshow(cv2.imread(gt))
        plt.savefig("example_images/preds_vs_ground_truth_box{}.png".format(bs), bbox_inches='tight')
        plt.show()
