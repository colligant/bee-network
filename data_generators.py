import numpy as np
import pdb
from matplotlib.pyplot import imread, imshow, show
from tensorflow.keras.utils import Sequence
from glob import glob
from cv2 import resize
from os.path import join

def _load_image(f, grayscale=False):
    im = imread(f).astype(np.uint8)
    return im/np.max(im)

class DataGenerator(Sequence):
#class DataGenerator():

    def __init__(self, data_directory, batch_size, resize=()):
        # resize should be tuple of (x_resize_factor, y_resize_factor)
        self.masks = sorted(glob(join(data_directory, 'masks', '*png')))
        self.images = sorted(glob(join(data_directory, 'images', '*jpg')))
        if len(self.masks) != len(self.images):
            raise ValueError('expected number of labels to equal number of \
                    images')
        self.n_instances = len(self.images)
        self.batch_size = batch_size
        self.resize = resize

    def __getitem__(self, idx):
        labels = self.masks[self.batch_size*idx:self.batch_size*(idx+1)]
        images = self.images[self.batch_size*idx:self.batch_size*(idx+1)]
        images = [_load_image(f) for f in images]
        labels = [np.round(_load_image(f)) for f in labels]
        if len(self.resize):
            for i in range(len(images)):
                images[i] = resize(images[i], (0,0), fx=self.resize[0],
                        fy=self.resize[1]) 
                # resize uses some sort of interpolation, meaning
                # values of inputs get changed. For labels,
                # the values need to be 0 or 1.
                labels[i] = np.round(resize(labels[i], (0,0),
                    fx=self.resize[0], fy=self.resize[1]))

        return np.asarray(images), np.expand_dims(np.asarray(labels), -1)

    def on_epoch_end(self):
        # shuffle data
        indices = np.random.choice(np.arange(self.n_instances),
                self.n_instances, replace=False)
        self.masks = list(np.asarray(self.masks)[indices])
        self.images = list(np.asarray(self.images)[indices])

    def __len__(self):
        return int(np.ceil(self.n_instances // self.batch_size))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dg = DataGenerator('train', 1, (0.25, 0.25))
    for i, m in dg:
        plt.figure()
        plt.imshow(np.squeeze(m))
        plt.colorbar()
        plt.figure()
        plt.imshow(np.squeeze(i))
        plt.colorbar()
        plt.show()
        break

