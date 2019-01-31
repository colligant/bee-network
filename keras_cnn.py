import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from data_utils import generate_labels_and_features, normalize_image 

def eval_image(path, model, kernel_size):
   im = cv2.imread(path)
   im = normalize_image(im)
   positive_x = [] 
   positive_y = [] 
   first = True
   ofs = kernel_size // 2
   threshold =  0.90
   for i in range(kernel_size, im.shape[0]):
       sub_imgs = np.zeros((im.shape[1]-kernel_size, kernel_size, kernel_size, 3))
       k = 0
       coord_x = []
       coord_y = []
       for j in range(kernel_size, im.shape[1]):
           sub_img = im[i-kernel_size:i, j-kernel_size:j, :] 
           sub_imgs[k, :, :, :] = sub_img
           k += 1
           coord_x.append(i-ofs)
           coord_y.append(j-ofs)

       result = model.predict(sub_imgs)
       result = result.astype(np.float32) 
       result = np.where(result[:, 1] > threshold)[0] # this should be 1-d.
       if first:
           print(result.shape)
           first = False
       if len(result) > 0:
           coord_x = np.asarray(coord_x)
           coord_y = np.asarray(coord_y)
           positive_x += list(coord_x[result])
           positive_y += list(coord_y[result])

       print(float(i)/im.shape[0])

   return positive_x, positive_y, cv2.imread(path) 

path = '/home/thomas/bee-network/image-labelling-tool/images/'
kernel_size = 71 
features, labels = generate_labels_and_features(path, kernel_size)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, labels,
        test_size=0.1, random_state=42)
# 
model = tf.keras.Sequential()
# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
    input_shape=(kernel_size,kernel_size,3))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
# Take a look at the model summary
model.summary()

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=14,
         validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])
qq = 'Day 1 Blank VS Scented328.jpg' 
jj = qq[:-4]
im_path = path + qq 
x, y, image = eval_image(im_path, model, kernel_size)
import matplotlib.pyplot as plt
plt.plot(y, x, "ro", ms=1)
plt.imshow(image)
#plt.savefig(jj + "xy.png")
plt.show()

