import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from data_generators import DataGenerator

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

    train_generator = DataGenerator('train', 2)
    test_generator = DataGenerator('test', 2)

    tb = TensorBoard()

    model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])


    model.fit_generator(train_generator,
            validation_data=test_generator,
            epochs=30,
            use_multiprocessing=True,
            )
    model.save("trained.h5")

    import numpy as np
    test_generator = DataGenerator('test', 1)
    for i, m in test_generator:
        fig, ax = plt.subplots(ncols=2, figsize=(18, 12))
        ax[0].imshow(i[0])
        ax[0].imshow(np.squeeze(m[0]), alpha=0.1)
        ax[1].imshow(np.squeeze(model.predict(i)))
        plt.show()





