import glob
import numpy as np
import matplotlib.pyplot as plt

from skimage import measure

def get_contours(stack, threshold=1, sz_contour=40):

    ls = []
    contours = measure.find_contours(stack, threshold)
    for contour in contours:
        a = len(contour[:, 1])
        b = len(contour[:, 0])
        if a > sz_contour and b > sz_contour:
            ls.append(contour)
    return ls 

def concat_preds(path, th=0.99):

    k_sizes = [11, 15]
    stack = np.zeros((1080, 1920))
    for i, ks in enumerate(k_sizes):
        mask = np.load(path + "mask_kernel{}.npy".format(ks))
        image = np.load(path + "image_kernel{}.npy".format(ks))
        mask[mask >= th] = 1
        mask[mask < th] = 0
        stack += mask
    return stack, image

if __name__ == '__main__':

    path = 'im_and_p_arrays/'

    for di in glob.glob(path + "*"):
        if not di.endswith("11_15"):
            print(di)
            di += '/'
            try:
                stack, image = concat_preds(di)
            except IOError as e:
                continue
            contours = get_contours(stack) 
            fig, ax = plt.subplots(figsize=(9,7), dpi=800)
            k = 0
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], 'r-', ms=2)
                k+=1
            plt.suptitle("Number of bees detected: {}".format(k))
            ax.imshow(image, alpha=0.99)
            ax.set_aspect('auto')
            plt.savefig("counted_images/counted_{}_k_11_15".format(di[-4:-1]))
            plt.close()

