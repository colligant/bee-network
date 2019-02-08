import os
import json
import matplotlib.pyplot as plt
import warnings
import numpy as np
import glob
import cv2
from sklearn.preprocessing import StandardScaler
from skimage.draw import polygon

def not_bee_mask(image, image_mask, n_instances, box_size):
    n = 0 
    instances = []
    while n <= n_instances: #ensure class balance
        x_rand = np.random.randint(box_size, image.shape[0]-box_size)
        y_rand = np.random.randint(box_size, image.shape[1]-box_size)
        if image_mask[x_rand, y_rand] != 1: # TODO: Make sure the neighborhood is not a bee.
            try:
                if box_size == 0:
                    image_mask[x_rand, y_rand] = 0
                    n += 1
                else:
                    image_mask[x_rand-box_size:x_rand+box_size,
                        y_rand-box_size:y_rand+box_size] = 0
                    n += box_size*2 
            except IndexError as e:
                continue
    return image_mask

def generate_not_bees(image, image_mask, num_points, kernel_size):
    ofs = kernel_size // 2
    n = 0 
    instances = []
    while n <= num_points: #ensure class balance
        x_rand = np.random.randint(0, image.shape[0])
        y_rand = np.random.randint(0, image.shape[1])
        if image_mask[x_rand, y_rand] != 1:
            try:
                sub_img = image[x_rand-ofs:x_rand+1+ofs, y_rand-ofs:y_rand+1+ofs,:] 
                # this line was returning arrays of weird sizes.
                if sub_img.shape[0] == kernel_size and sub_img.shape[1] == kernel_size:
                    instances.append(sub_img)
                    n += 1
            except IndexError as e:
                continue
    return instances

def normalize_image(im):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tmp = np.zeros((im.shape))
        mask = np.zeros((im.shape[0], im.shape[1]))
        scaler = StandardScaler() 
        scaler.fit(im[:, :, 0])
        tmp[:, :, 0] = scaler.transform(im[:, :, 0])
        scaler2 = StandardScaler()
        scaler2.fit(im[:, :, 1])
        tmp[:, :, 1] = scaler2.transform(im[:, :, 1])
        scaler3 = StandardScaler()
        scaler3.fit(im[:, :, 2])
        tmp[:, :, 2] = scaler3.transform(im[:, :, 2])
        # Z-NORMALIZE! How? Unravel the image and 
        # take the mean and stddev? Probs. 
    return tmp 

def generate_class_mask(f_polygons, f_image, box_size=0): 
    im = cv2.imread(f_image)
    mask = np.ones((im.shape[0], im.shape[1]))*-1
    im = normalize_image(im) 
    with open(f_polygons, "r") as f:
      js = json.load(f)
    n_instances = 0
    for ll in js['labels']:
      x = []
      y = []
      for i in ll['vertices']:
          x.append(i['x'])
          y.append(i['y'])
      r, c = polygon(x, y, shape=im.shape) # row, column
      mask[c, r] = 1
      n_instances += len(r)

    if n_instances > 1:
        mask = not_bee_mask(im, mask, n_instances=n_instances, box_size=box_size)
        #mask[mask == -1] = 0
        # mask[mask == -1] = 
        # print(2*n_instances, "total training points")
        return mask, im
    else:
        return None, None


def generate_bees(f_polygons, f_image, kernel_size):
    # Do I want to do a pixel-wise 
    # classifier? Or, say, one training instance per bee?
    # would a FCNN on bee images work?
    ofs = kernel_size // 2
    im = cv2.imread(f_image)
    mask = np.zeros((im.shape[0], im.shape[1]))
    im = normalize_image(im) 
    with open(f_polygons, "r") as f:
      js = json.load(f)
    for ll in js['labels']:
      x = []
      y = []
      for i in ll['vertices']:
          x.append(i['x'])
          y.append(i['y'])
      r, c = polygon(x, y, shape=im.shape) # row, column
      mask[r, c] = 1
      instances = []
      for rr, cc in zip(r, c):
          try:
              sub_img = im[cc-ofs:cc+ofs+1,rr-ofs:rr+ofs+1, :]  # I think this line
              if sub_img.shape[0] == kernel_size and sub_img.shape[1] == kernel_size:
                  instances.append(sub_img)
          except IndexError as e:
              continue

    return instances, mask, im

def generate_labels_and_features(path, kernel_size):
    bees = []
    not_bees = []

    for f in glob.glob(path + "*.json"):
        jpg = f[:-13] + ".jpg"
        features, mask, image = generate_bees(f, jpg, kernel_size)
        bees += features
        neg_features = generate_not_bees(image, mask, len(features), kernel_size)
        not_bees += neg_features

    features = bees + not_bees
    u = features[0].shape
    ret = np.zeros((len(features), u[0], u[1], u[2]))
    for i, e in enumerate(features):
    # print(i, e.shape, ret.shape)
        ret[i, :, :, :] = e
    labels = [1]*len(bees) + [0]*len(not_bees)
    labels = make_one_hot(labels, 2)
    return np.asarray(features), np.asarray(labels)

def make_one_hot(labels, n_classes):

    ret = np.zeros((len(labels), n_classes))
    for i, e in enumerate(labels):
        ret[i, e] = 1
    return ret


if __name__ == '__main__':
    path = '/home/thomas/bee-network/for_bees/Blank VS Scented/B VS S Day 1/Frames JPG/'
 
    for f in glob.glob(path + "*.json"):
        print(f[-17:-13])
        jpg = f[:-13] + ".jpg"
        mask, image = generate_class_mask(f, jpg)
        print(image.shape)
        break
        if mask is not None:
            mask[mask == -1] = np.nan
            plt.title(os.path.basename(f))
            plt.imshow(image)
            plt.imshow(mask, alpha=0.8)
            plt.show()
 
