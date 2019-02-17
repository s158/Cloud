import scipy.io
mat = scipy.io.loadmat('Nx.mat')
Nx = mat['Nx']
mat = scipy.io.loadmat('Ny.mat')
Ny = mat['Ny']
mat = scipy.io.loadmat('img.mat')
img = mat['Images']

import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([
    tf.keras.layers.Convolution2D(filters = 8 , kernel_size = (5,5) , activation=tf.nn.relu ,
                                  padding='same', input_shape = (450,450,1)),
    tf.keras.layers.Convolution2D(filters = 1 , kernel_size = 5 , activation=tf.nn.relu ,
                                  padding='same')
])

model.compile(optimizer='adam',
              loss='mean_squared_error')

img = img.reshape((-1, 450, 450, 1))
Nx = Nx.reshape((-1,450,450,1))
#np.reshape(img,[-1,450,450,1])
#np.reshape(Nx,[-1,450,450,1])
model.fit(img, Nx, epochs=5)
