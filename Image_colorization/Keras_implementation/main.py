import numpy as np

from IPython.display import display, Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, UpSampling2D, Layer, Input # type: ignore
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.utils import img_to_array, load_img # type: ignore
from skimage.color import lab2rgb, rgb2lab
from skimage import color
import keras as keras
import tensorflow as tf
import glob
import cv2 as cv2
import os
import pdb


folder_path = 'Image_colorization/Keras_implementation/Data/Black_White'
images1 = []
for img in os.listdir(folder_path):
    img=folder_path+"/"+img
    img = load_img(img, target_size=(100,100))
    img = img_to_array(img) / 255
    X = color.rgb2gray(img)
    images1.append(X)

folder_path = 'Image_colorization/Keras_implementation/Data/colored'
images2 = []
for img in os.listdir(folder_path):
    img=folder_path+'/'+img
    img = load_img(img, target_size=(100,100)) 
    img = img_to_array(img)/ 255
    lab_image = rgb2lab(img)
    lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
    Y = lab_image_norm[:,:,1:]

    images2.append(Y)

X = np.array(images1)
Y = np.array(images2)

class ReshapeLayer(Layer):
    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, self.target_shape)

# Custom layer for tf.image.resize
class ResizeLayer(Layer):
    def __init__(self, target_size, **kwargs):
        super().__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size)

# Model definition
x1 = Input(shape=(None, None, 1))

x2 = Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(x1)
x3 = Conv2D(16, (3, 3), activation='relu', padding='same')(x2)
x4 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(x3)
x5 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
x6 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x5)
x7 = UpSampling2D((2, 2))(x6)
x8 = Conv2D(32, (3, 3), activation='relu', padding='same')(x7)
x9 = UpSampling2D((2, 2))(x8)
x10 = Conv2D(16, (3, 3), activation='relu', padding='same')(x9)
x11 = UpSampling2D((2, 2))(x10)
x12 = Conv2D(2, (3, 3), activation='sigmoid', padding='same')(x11)

# Wrap TensorFlow operations in layers
x12 = ReshapeLayer((-1, 104, 104, 2))(x12)
x12 = ResizeLayer([100, 100])(x12)
x12 = ReshapeLayer((1, 100, 100, 2))(x12)

# Finish model
model = Model(inputs=x1, outputs=x12)

model.compile(optimizer='rmsprop', loss='mse')

model.fit(X, Y, batch_size=1, epochs=400, verbose=1)

model.evaluate(X, Y, batch_size=1)

# Load and preprocess image
img_path = 'Image_colorization/Keras_implementation/Data/Test/Gray_33.jpg'

img = load_img(img_path, target_size=(100, 100), color_mode="grayscale")
img = img_to_array(img) / 255.0
ss = img.shape

X = np.expand_dims(img, axis=-1)
X = np.reshape(X, (1, 100, 100, 1))

# Predict and post-process
output = model.predict(X)
output = np.reshape(output, (100, 100, 2))
output = cv2.resize(output, (ss[1], ss[0]))
AB_img = output

outputLAB = np.zeros((ss[0], ss[1], 3))
outputLAB[:, :, 0] = np.reshape(img, (ss[0], ss[1]))
outputLAB[:, :, 1:] = AB_img
outputLAB = (outputLAB * [100, 255, 255]) - [0, 128, 128]

rgb_image = lab2rgb(outputLAB)

# Visualize
plt.imshow(rgb_image)
plt.axis('off')
plt.show()

'''
x1 = keras.Input(shape=(None, None, 1))

x2 = Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(x1)
x3 = Conv2D(16, (3, 3), activation='relu', padding='same')(x2)
x4 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(x3)
x5 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
x6 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x5)
x7 = UpSampling2D((2, 2))(x6)
x8 = Conv2D(32, (3, 3), activation='relu', padding='same')(x7)
x9 = UpSampling2D((2, 2))(x8)
x10 = Conv2D(16, (3, 3), activation='relu', padding='same')(x9)
x11 = UpSampling2D((2, 2))(x10)
x12 = Conv2D(2, (3,3), activation='sigmoid', padding='same')(x11)

x12=tf.reshape(x12,(104,104,2))
x12 = tf.image.resize(x12,[100, 100])
x12=tf.reshape(x12,(1,100, 100,2))

# Finish model
model = keras.Model(x1, x12)

model.compile(optimizer='rmsprop', loss='mse')
model.fit(X,Y, batch_size=1, epochs=400, verbose=1)

model.evaluate(X, Y, batch_size=1)

folder_path='Image_colorization/Keras_implementation/Data/Test' 
img='Gray_33.jpg'
img=folder_path+img


img = load_img(img, target_size=(100,100),color_mode = "grayscale") 
img = img_to_array(img)/ 255
ss=img.shape

X = np.array(img)
X = np.expand_dims(X, axis=2)
X=np.reshape(X,(1,100,100,1))
output = model.predict(X)
output=np.reshape(output,(100,100,2))
output=cv2.resize(output,(ss[1],ss[0]))
AB_img = output
outputLAB = np.zeros((ss[0],ss[1], 3))
img=np.reshape(img,(100,100))
outputLAB[:,:,0]=img
outputLAB[:,:,1:]=AB_img
outputLAB = (outputLAB * [100, 255, 255]) - [0, 128, 128]
rgb_image = lab2rgb(outputLAB)

import matplotlib.pyplot as plt

imshow(rgb_image)
plt.show()
'''