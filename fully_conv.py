import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from glob import glob
from data_utils_polygons import generate_class_mask, normalize_image, make_bee_squares
from skimage import transform, util
from tensorflow.keras.layers import (Conv2D, Input, MaxPooling2D, Conv2DTranspose, 
Concatenate, Dropout, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

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

def fcnn_functional(image_shape, n_classes):

    x = Input(image_shape)

    c1 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x)
    c1 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(c1)
    mp1 = MaxPooling2D(pool_size=2, strides=(2, 2))(c1)

    c2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(mp1)
    c2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(c2)
    mp2 = MaxPooling2D(pool_size=2, strides=(2, 2))(c2)

    c3 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(mp2)
    c3 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(c3)
    mp3 = MaxPooling2D(pool_size=2, strides=(2, 2))(c3)
    
    last_conv = Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(mp3)

    u1 = UpSampling2D(size=(2, 2))(last_conv)
    u1 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(u1)
    u1 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(u1)

    u1_c3 = Concatenate()([c3, u1])

    u2 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(u1_c3)
    u2 = UpSampling2D(size=(2, 2))(u2)
    u2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(u2)
    u2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(u2)

    u2_c2 = Concatenate()([u2, c2])

    c4 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(u2_c2)
    u3 = UpSampling2D(size=(2, 2))(c4)
    u3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(u3)

    u3_c1 = Concatenate()([u3, c1])

    c5 = Conv2D(filters=n_classes, kernel_size=(3,3), activation='softmax', padding='same')(u3_c1)

    model = Model(inputs=x, outputs=c5) 
    return model


def fcnn_model(image_shape, n_classes):
    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
        input_shape=image_shape, data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=n_classes, kernel_size=2, padding='same',
        activation='sigmoid')) # 1x1 convolutions for pixel-wise prediciton.
    # Take a look at the model summary
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
    '''Randomly augments an image.'''
    if np.random.randint(2):
        deg = np.random.uniform(-25, 25)
        image = rotation(image, deg)
        class_mask = rotation(class_mask, deg)
    if np.random.randint(2):
        image = random_noise(image)
    if np.random.randint(2):
        image = h_flip(image)
        class_mask = h_flip(class_mask)
    if np.random.randint(2):
        image = np.flipud(image)
        class_mask = np.flipud(class_mask)
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


def generate(image_directory, box_size):
    from random import shuffle
    while True:
        fnames = [f for f in glob(image_directory + "*.json")]
        shuffle(fnames)
        for f in fnames:
            #shuffle this data
            jpg = f[:-13] + ".jpg"
            outa = []
            outb = []
            batch_size = 100
            i = 0
            for data, mask in make_bee_squares(f, jpg):
                # X = np.expand_dims(input_image, axis=0)
                # y = np.expand_dims(class_mask, axis=0)
                if i < batch_size:
                    outa.append(data)
                    outb.append(mask)
                    i += 1
                if i == batch_size:
                    break
            if not len(outa):
                continue
            yield np.asarray(outa), np.asarray(outb)


def create_model(image_shape, n_classes, learning_rate=1e-6):
    model = fcnn_model(image_shape, n_classes)
    #model = unet()
    model.compile(loss='binary_crossentropy',
                 optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
                 metrics=['accuracy'])
    return model

def train_model(train_directory, test_directory, learning_rate, image_shape, box_size=6, epochs=15):
    n_classes = 2
    model = create_model(image_shape, n_classes, learning_rate)
    tb = TensorBoard(log_dir='graphs/')
    n_augmented = 0
    train_generator = generate(train_directory, box_size)
    test_generator = generate(test_directory, box_size)
    model.fit_generator(train_generator, 
            steps_per_epoch=50, 
            epochs=epochs,
            verbose=1,
            callbacks=[tb],
            use_multiprocessing=True)
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
    out = np.reshape(out, (1080, 1920, 2))
    return out

if __name__ == '__main__':
    from sklearn.metrics import confusion_matrix
    train = '/home/thomas/bee-network/for_bees/Blank VS Scented/B VS S Day 1/Frames JPG/'
    test = '/home/thomas/bee-network/for_bees/Blank VS Scented/B VS S Day 1/Frames JPG/ground_truth/'
    shape = (None, None, 3)
    box_size = 10 
    model_path = 'models/fcnn_functional.h5'
    epochs = 100 
    lr = 1e-3
    if os.path.isfile(model_path): 
        model = train_model(train, test, lr, shape, box_size, epochs=epochs)
        model.save(model_path)
    else:
        model = tf.keras.models.load_model(model_path,
                custom_objects={'custom_objective':custom_objective})
    
    for f in glob(test + "*.jpg"):
        out = evaluate_image(f, model)
        plt.imshow(out[:, :, 1])
        plt.colorbar()
        plt.show()
