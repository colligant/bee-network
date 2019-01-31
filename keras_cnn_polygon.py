import glob
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from data_utils_polygons import generate_labels_and_features, normalize_image 
from count_bees import get_contours
from tqdm import tqdm
from multiprocessing import Pool

def eval_image_heatmap(path, model, kernel_size):
   im = cv2.imread(path) # plt.imread only has png support
   im = normalize_image(im)
   mask = np.zeros((im.shape[0], im.shape[1]))
   ofs = kernel_size // 2
   step_size = 1
   for i in range(kernel_size, im.shape[0], step_size):
       sub_imgs = np.zeros((im.shape[1]-kernel_size, kernel_size, kernel_size, 3))
       k = 0
       for j in range(kernel_size, im.shape[1], step_size):
           sub_img = im[i-kernel_size:i, j-kernel_size:j, :] 
           sub_imgs[k, :, :, :] = sub_img
           k += step_size

       result = model.predict(sub_imgs)
       result = result.astype(np.float32) 
       result = result[:, 1]
       mask[i-ofs, kernel_size - ofs: -(kernel_size-ofs-1)] = result

   return mask, cv2.imread(path) 


def keras_model(kernel_size):
    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
        input_shape=(kernel_size, kernel_size, 3))) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    # Take a look at the model summary
    model.summary()
    return model


def train_model(kernel_size, path):
    features, labels = generate_labels_and_features(path, kernel_size)

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(features, labels,
            test_size=0.1, random_state=42)

    model = keras_model(kernel_size)
    model.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    model.fit(x_train,
             y_train,
             batch_size=128,
             epochs=10,
             validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n', 'Test accuracy:', score[1])
    return model


def plot_contours(contours, ax):
    ''' Returns num_contours as well '''
    k = 0
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], 'r-', ms=2)
        k+=1
    return k

def run_model(kernel_size, model_path, image_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    mask, image = eval_image_heatmap(image_path, model, kernel_size)
    return mask

def get_model_paths(kernel_sizes, base_path):
    ls = []
    for f in glob.glob(base_path + "*.h5"): 
        for ks in kernel_sizes:
            model_path = 'models/model_kernel{}.h5'.format(ks)
            if not os.path.isfile(model_path):
                print("model not present for kernel size {}".format(cw))
            else:
                ls.append(model_path)
    return ls

def concat_preds(arr_list, th=0.95):

    stack = np.zeros(arr_list[0].shape)
    for i, mask in enumerate(arr_list):
        mask[mask >= th] = 1
        mask[mask < th] = 0
        stack += mask
    return stack


if __name__ == '__main__':
    kernel_sizes = [11, 15]
    image_paths = []
    model_dir = 'models/'
    model_paths = get_model_paths(kernel_sizes, model_dir)
    path = '/home/thomas/bee-network/for_bees/bees_polygons/Frames JPG/'
    for f in glob.glob(path + "*.jpg"):
        image_paths = [f]*len(kernel_sizes)
        with Pool(len(kernel_sizes)) as pool:
            results = pool.starmap(run_model, zip(kernel_sizes, model_paths, image_paths))
        
        out_mask = concat_preds(results)
        out_image = cv2.imread(image_paths[0])

        contours = get_contours(out_mask, threshold=1)
        fig, ax = plt.subplots(ncols=1, figsize=(9, 7))
        bees = plot_contours(contours, ax)
        out_mask[out_mask == 0] = np.nan
        ax.imshow(out_image)
        ax.imshow(out_mask, alpha=0.5)
        ax.set_title("Prob. heatmap")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.suptitle("Number of bees: {}".format(bees))
        #plt.savefig("evaluated_images/step_size_16.png")
        plt.show()

if __name__ != '__main__':
    import tensorflow as tf
    ''' Single threaded evaluation '''
    ls = [15, 31]
    iii = 0
    path = '/home/thomas/bee-network/for_bees/bees_polygons/Frames JPG/'
    threshold = 0.9
    for f in glob.glob(path + "*.jpg"):
        if not os.path.isfile(f[:-4] + "__.json"):
            out_mask = None
            out_image = None
            for dx, i in enumerate(ls):
                print("Evaluating {} ({}/{} kernel sizes done)".format(f, dx, len(ls)))
                kernel_size = i
                model_path = 'models/model_kernel{}.h5'.format(kernel_size)
                if not os.path.isfile(model_path):
                    model = train_model(kernel_size, path)
                    model.save(model_path)
                else:
                    model = tf.keras.models.load_model(model_path)
                num = f[-8:-4]
                jpg = f 
                # not present in training data. 
                # im_path = path + jpg 
                mask, image = eval_image_heatmap(jpg, model, kernel_size)
                if not isinstance(out_mask, np.ndarray):
                    mask[mask < threshold] = 0
                    mask[mask >= threshold] = 1
                    out_mask = mask
                    out_image = image
                else:
                    mask[mask < threshold] = 0
                    mask[mask >= threshold] = 1
                    out_mask += mask

            # out_mask[out_mask <kkkkkkkkkkkk threshold] = np.nan
            contours = get_contours(out_mask, threshold=1)
            fig, ax = plt.subplots(ncols=1, figsize=(9, 7))
            bees = plot_contours(contours, ax)
            out_mask[out_mask == 0] = np.nan
            ax.imshow(out_image)
            ax.imshow(out_mask, alpha=0.5)
            ax.set_title("Prob. heatmap")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.suptitle("Number of bees: {}".format(bees))
            #plt.savefig("evaluated_images/step_size_16.png")
            plt.show()

                
                
                
                
                
                
