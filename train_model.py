import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from data_generators import DataGenerator

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed



def simple_fcnn(image_shape):
    # gotchas that i worked through:
    # padding='same'
    # labels not having an extra dimension to 
    # correspond to 
    inp = Input(image_shape)
    c1 = Conv2D(8, 3, padding='same', activation='relu')(inp)
    m1 = MaxPooling2D()(c1)
    c2 = Conv2D(16, 3, padding='same', activation='relu')(m1)
    c4 = Conv2D(16, 3, padding='same', activation='relu')(c2)
    u2 = UpSampling2D()(c4)
    c5 = Conv2D(1, 3, padding='same', activation='sigmoid')(u2)

    return Model(inputs=inp, outputs=c5)


if __name__ == '__main__':

    model = simple_fcnn((1080, 1920, 3))

    train_generator = DataGenerator('subset', 2)
    test_generator = DataGenerator('subset', 2)

    tb = TensorBoard()

    model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model_path = 'subset_trained.h5'
    if not os.path.isfile(model_path):
        model.fit_generator(train_generator,
                validation_data=test_generator,
                epochs=30,
                use_multiprocessing=True,
                )
        model.save(model_path)
    else:
        custom_objects = {'binary_focal_loss_fixed':binary_focal_loss()}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # test_generator = DataGenerator('train', 1, resize=(0.5, 0.5))
    test_generator = DataGenerator('train', 1, resize=(1, 1))
    for j, (i, m) in enumerate(test_generator):
        fig, ax = plt.subplots()
        #ax.imshow(i[0])
        preds = np.squeeze(model.predict(i.astype(np.float16)))
        ax.imshow(preds)
        plt.show()
        if j > 3:
            break
