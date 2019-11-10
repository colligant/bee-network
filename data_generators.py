import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from cv2 import resize
from os.path import join

def _load_image(f, grayscale=False):
    return plt.imread(f)

# class DataGenerator(Sequence):
class DataGenerator():

    def __init__(self, data_directory, batch_size):

        self.masks = sorted(glob(join(data_directory, 'masks', '*png')))
        self.images = sorted(glob(join(data_directory, 'images', '*jpg')))
        if len(self.masks) != len(self.images):
            raise ValueError('expected length of labels to equal lengths of \
                    images')
        self.n_instances = len(self.images)
        self.batch_size = batch_size

    def __getitem__(self, idx):
        labels = self.masks[idx:self.batch_size*(idx+1)]
        images = self.images[idx:self.batch_size*(idx+1)]
        images = [_load_image(f) for f in images]
        labels = [_load_image(f) for f in labels]
        return np.asarray(images), np.asarray(labels)

    def on_epoch_end(self, epoch):
        indices = np.random.choice(np.arange(self.n_instances),
                self.n_instances, replace=False)
        self.masks = list(np.asarray(self.masks)[indices])
        self.images = list(np.asarray(self.images)[indices])

    def __len__(self):
        return np.ceil(self.n_instances // self.batch_size)


if __name__ == '__main__':
    pass
