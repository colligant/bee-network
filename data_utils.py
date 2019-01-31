from sklearn.preprocessing import StandardScaler
import json
import numpy as np
import glob
import cv2 

def normalize_image(im):
    tmp = np.zeros((im.shape))
    mask = np.zeros((im.shape[0], im.shape[1]))
    scaler = StandardScaler()
    scaler.fit(im[:, :, 0])
    tmp[:, :, 0] = scaler.transform(im[:, :, 0])
    scaler2 = StandardScaler()
    scaler2.fit(im[:, :, 1])
    tmp[:, :, 1] = scaler2.transform(im[:, :, 1])
    scaler3 = StandardScaler()
    scaler3.fit(im[:, :, 2]) tmp[:, :, 2] = scaler3.transform(im[:, :, 2])
    # Z-NORMALIZE! How? Unravel the image and 
    # take the mean and stddev? Probs. 
    return tmp 


def generate_bees(f_polygons, f_image, kernel_size):
    # Do I want to do a pixel-wise 
    # classifier? Or, say, one training instance per bee?
    # would an FCNN on bee images work?
    ofs = kernel_size // 2
    im = cv2.imread(f_image)
    mask = np.zeros((im.shape[0], im.shape[1]))
    im = normalize_image(im) 
    with open(f_polygons, "r") as f:
      js = json.load(f)
    not_bees = []
    bees = []
    for ll in js['labels']:
        if ll['label_class'] is not None:
            rr = ll['position']['x']
            cc = ll['position']['y']
            rr = int(rr)
            cc = int(cc)
            try:
                sub_img = im[cc-ofs:cc+ofs+1,rr-ofs:rr+ofs+1, :]  
                if sub_img.shape[0] == kernel_size and sub_img.shape[1] == kernel_size:
                    not_bees.append(sub_img)
            except IndexError as e:
                print("except")
                continue
        else:
            rr = ll['position']['x']
            cc = ll['position']['y']
            rr = int(rr)
            cc = int(cc)
            # import matplotlib.pyplot as plt
            # imm = cv2.imread(f_image)
            # fig, axs = plt.subplots(ncols=2)
            # axs[0].imshow(imm)
            # axs[0].plot(rr, cc, 'ro')
            # # THIS WAS WRONG!
            # axs[1].imshow(imm[cc-ofs:cc+ofs+1,rr-ofs:rr+ofs+1, :])
            # plt.show()
            try:
                sub_img = im[cc-ofs:cc+ofs+1,rr-ofs:rr+ofs+1, :]  
                #plt.imshow(sub_img)
                #plt.show()
                if sub_img.shape[0] == kernel_size and sub_img.shape[1] == kernel_size:
                    bees.append(sub_img)
            except IndexError as e:
                print("except")
                continue
    return bees, not_bees 

def generate_labels_and_features(path, kernel_size):
    bees = []
    not_bees = []

    for f in glob.glob(path + "*.json"):
        jpg = f[:-13] + ".jpg"
        features, not_features = generate_bees(f, jpg, kernel_size)
        bees += features
        not_bees += not_features

    features = bees + not_bees
    u = features[0].shape
    ret = np.zeros((len(features), u[0], u[1], u[2]))
    for i, e in enumerate(features):
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
    pass 
    path = '/home/thomas/bee-network/image-labelling-tool/images/'
    feat, lab = generate_labels_and_features(path, 101)
