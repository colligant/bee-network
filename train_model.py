import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

from sys import stdout
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, 
        UpSampling2D, Concatenate)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

from data_generators import DataGenerator

def precision_and_recall_from_generator(generator, model, n_classes=2):
    cmat = np.zeros((n_classes, n_classes))
    for j, (i, m) in enumerate(test_generator):
        stdout.write('{} / {}\r'.format(j, len(generator)))
        preds = np.squeeze(model.predict(i.astype(np.float16)))
        y_pred = np.round(preds).astype(np.uint)
        y_true = np.squeeze(m).astype(np.uint)
        cmat += confusion_matrix(y_true.ravel(), y_pred.ravel(), labels=[0, 1])
    print()
    return cmat


def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage: model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
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
    inp = Input(image_shape)
    c1 = Conv2D(16, 3, padding='same', activation='relu')(inp)
    c1 = Conv2D(32, 3, padding='same', activation='relu')(c1)
    m1 = MaxPooling2D()(c1)
    c2 = Conv2D(64, 3, padding='same', activation='relu')(m1)
    c3 = Conv2D(64, 3, padding='same', activation='relu')(c2)
    u2 = UpSampling2D()(c3)
    concat = Concatenate()([c1, u2])
    c4 = Conv2D(32, 3, padding='same', activation='relu')(concat)
    c4 = Conv2D(16, 3, padding='same', activation='relu')(c4)
    c5 = Conv2D(1, 3, padding='same', activation='sigmoid')(c4)
    return Model(inputs=inp, outputs=c5)


def weighted_xen(y_true, y_pred):
    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred,
            pos_weight=1.0)


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


if __name__ == '__main__':

    model = simple_fcnn((540, 960, 3))

    resize = (0.5, 0.5)
    train_generator = DataGenerator('train_data', 4, resize=resize)
    test_generator = DataGenerator('test_data', 2, resize=resize)
    tb = TensorBoard()
    loss_func = binary_focal_loss()
    model.compile(Adam(1e-3), loss=loss_func, metrics=['accuracy'])
    model_path = 'models/half_image.h5'
    if not os.path.isfile(model_path):
        model.fit_generator(train_generator,
                epochs=40,
                #validation_data=test_generator,
                verbose=1,
                use_multiprocessing=True,
                )
        model.save(model_path)
    else:
        custom_objects = {'loss':loss,
                'binary_focal_loss_fixed':binary_focal_loss(),
                'dice_coef_loss':dice_coef_loss}

        model = tf.keras.models.load_model(model_path,
                custom_objects=custom_objects)

    test_generator = DataGenerator('test_data', 1, resize=resize)
    conf_mat = precision_and_recall_from_generator(test_generator, model)
    print(conf_mat)
    precision = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    recall = conf_mat[1, 1] / (conf_mat[0, 1] + conf_mat[1, 1])
    print('precision: {:.3f}, recall: {:.3f}'.format(precision, recall))
    # for j, (i, m) in enumerate(test_generator):
    #     fig, ax = plt.subplots(ncols=2)
    #     ax[0].imshow(i[0])
    #     preds = np.squeeze(model.predict(i.astype(np.float16)))
    #     ax[1].imshow(np.round(preds))
    #     plt.show()
    #     if input('continue?') == 'N':
    #         break
