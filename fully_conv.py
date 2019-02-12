import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
#tf.enable_eager_execution()
from glob import glob
from data_utils_polygons import generate_class_mask, normalize_image
from skimage import transform, util
from tensorflow.keras.layers import (Conv2D, Input, MaxPooling2D, Conv2DTranspose, 
Concatenate, Dropout, UpSampling2D)
from tensorflow.keras.models import Model
from keras.callbacks import TensorBoard

NO_DATA = 3
concatenate=Concatenate
def unet(input_size = (1080, 1920, 3)):
    inputs = Input(input_size)
    reshaped = tf.keras.layers.Cropping2D(cropping=((0, 8), (0, 0)))(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer =
            'he_normal')(reshaped)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate()([drop4,up6])
    conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate()([conv3,up7])
    conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate()([conv2,up8])
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate()([conv1,up9])
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(filters=2, kernel_size=1, activation = 'sigmoid')(conv9)
    out = tf.keras.layers.ZeroPadding2D(((0, 8), (0, 0)))(conv10)
    model = Model(inputs=inputs, outputs=out)
    #model.summary()

    return model 

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

def fcnn_functional(image_shape, n_classes):
    x = Input(image_shape)

    c1 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x)
    mp1 = MaxPooling2D(pool_size=2, strides=(2, 2))(c1)

    c2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(mp1)
    mp2 = MaxPooling2D(pool_size=2, strides=(2, 2))(c2)
    mp2 = Dropout(0.5)(mp2)

    c3 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(mp2)
    mp3 = MaxPooling2D(pool_size=2, strides=(2, 2))(c3)
    
    last_conv = Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(mp3)

    u1 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2))(last_conv)
    u1 = Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(u1)

    c3_pad = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)))(c3)
    u1_c3 = Concatenate()([c3_pad, u1])

    u2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(u1_c3)
    u2 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2))(u2)
    u2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(u2)

    c2_pad = tf.keras.layers.ZeroPadding2D(((0, 3), (0, 3)))(c2)
    u2_c2 = Concatenate()([u2, c2_pad])
    u2_c2 = Dropout(0.5)(u2_c2)

    c4 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(u2_c2)
    u3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2))(c4)
    u3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(u3)

    c1_pad = tf.keras.layers.ZeroPadding2D(((0, 7), (0, 7)))(c1)
    u3_c1 = Concatenate()([u3, c1_pad])

    c4 = Conv2D(filters=n_classes, kernel_size=(3,3), activation='softmax', padding='same')(u3_c1)

    c4 = tf.keras.layers.Cropping2D(cropping=((0, 7), (0, 7)))(c4)

    model = Model(inputs=x, outputs=c4) 
    model.summary()
    return model


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


def one_hot_encoding(class_mask, n_classes):
    '''Assumes classes range from 0 -> (n-1)'''
    shp = class_mask.shape
    out = np.ones((shp[0], shp[1], n_classes))*NO_DATA
    for i in range(n_classes):
        out[:, :, i][class_mask == i] = 1
    return out

def rotation(image, angle):
    return transform.rotate(image, angle, mode='constant', cval=NO_DATA)

def random_noise(image):
    return util.random_noise(image)

def h_flip(image):
    return image[:, ::-1]

def augment_data(image, class_mask):
    '''Randomly augments an image in a given way.'''
    if np.random.randint(2):
        deg = np.random.uniform(-25, 25)
        image = rotation(image, deg)
        class_mask = rotation(class_mask, deg)
    if np.random.randint(2):
        image = random_noise(image)
    if np.random.randint(2):
        image = h_flip(image)
        class_mask = h_flip(class_mask)
    return image, class_mask

def preprocess_training_data(image, class_mask, n_classes=2):
    ''' Have to reshape it such that there is 
        a batch size.'''
    input_image = image.astype(np.float32)
    class_mask[class_mask == -1] = NO_DATA 
    class_mask = class_mask.astype(np.int32)
    class_mask = one_hot_encoding(class_mask, n_classes) 
    in_class = np.zeros((1, 1080, 1920, n_classes))
    in_img = np.zeros((1, 1080, 1920, 3))
    in_class[0, :, :, :] = class_mask
    in_img[0, :, :, :] = input_image

    return in_img, in_class

def create_model(image_shape, n_classes):
    model = fcnn_functional(image_shape, n_classes)
    #model = unet()
    model.compile(loss=custom_objective,
                 optimizer='adam', 
                 metrics=['accuracy'])

    return model

def train_model(image_directory, image_shape, box_size, epochs=15):
    n_classes = 2
    model = create_model(image_shape, n_classes)
    tb = TensorBoard(log_dir='graphs/')
    n_augmented = 0
    for i in range(epochs):
        print("Epoch:", i)
        for j, f in enumerate(glob(image_directory + "*.json")):
            jpg = f[:-13] + ".jpg"
            class_mask, input_image = generate_class_mask(f, jpg, box_size=box_size)
            if class_mask is None:
                continue
            if np.random.randint(2): 
                print("Augmenting data")
                input_image, class_mask = augment_data(input_image, class_mask)
                # fig, axs = plt.subplots(ncols=2)
                # axs[0].imshow(input_image)
                # axs[1].imshow(class_mask)
                # plt.show()

            in_img, in_class = preprocess_training_data(input_image, class_mask)
            model.fit(in_img, in_class, epochs=1, verbose=1)

    return model

def evaluate_image(image_path, model, th=0.1):
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

def heatmap(image_path, model, th=0.6):
    im = cv2.imread(image_path)
    im = normalize_image(im)
    in_img = np.zeros((1, 1080, 1920, 3))
    in_img[0, :, :, :] = im
    start = time.time()
    out = model.predict(in_img)
    end = time.time()
    print("T_pred:", end-start)
    out = out[0:, :, :, :]
    #out = np.argmax(out, axis=3)
    out = out[:, :, 1]
    out = np.reshape(out, (1080, 1920))
    return out


def compute_iou(y_pred, y_true):
     ''' This is slow. '''
     y_pred = y_pred.flatten()
     y_true = y_true.flatten()
     current = confusion_matrix(y_true, y_pred, labels=[0, 1])
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     IoU = intersection / union.astype(np.float32)
     return np.mean(IoU)

if __name__ == '__main__':
    from sklearn.metrics import confusion_matrix
    path = '/home/thomas/bee-network/for_bees/Blank VS Scented/B VS S Day 1/Frames JPG/'
    shape = (1080, 1920, 3)
    box_size = 10 
    model_path = 'models/fcnn_functional.h5'

    if not os.path.isfile(model_path): 
        model = train_model(path, shape, box_size)
        model.save(model_path)
    else:
        model = tf.keras.models.load_model(model_path,
                custom_objects={'custom_objective':custom_objective})

    gt = 'ground_truth.jpg'
    json = 'ground_truth_labels.json'
    class_mask, in_img = generate_class_mask(json, gt)
    class_mask[class_mask == -1] = 0
    predictions = evaluate_image(gt, model)
    iou = compute_iou(predictions, class_mask)
    print("IoU: {}".format(iou))
    fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
    #hmap = heatmap(gt, model)
    ax[0].imshow(predictions)
    ax[0].set_title("Preds, box_size={}, IoU={:.3f}".format(box_size, iou))
    ax[1].imshow(cv2.imread(gt))
    #plt.savefig("example_images/preds_vs_ground_truth_box_size{}.png".format(box_size), bbox_inches='tight')
    plt.show()
