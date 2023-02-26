import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, Concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import save_model
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn import preprocessing
import gc

#collects unused data and deletes it
gc.enable()

#batch size, image width and image height in pixels
batch_size = 512
img_height = 30
img_width = 862

#change directory to get data
os.chdir('/localstorage/s1842521/')

temperature = np.load('processed_train_temperatures.npy')
pressure = np.load('pressure.npy')
parameters = np.load('processed_train_parameters.npy')
opacities = np.load('processed_train_opacities.npy')
temperatures_unchanged = np.load('temperature.npy')

#build the model
inputs = tf.keras.Input(shape=(30,862,2))

x = Conv2D(16, (3,10), activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(32, (3,10), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(64, (3,10), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Flatten()(x)

#input scaler values (parameters) into model
input_scalers = Input((3,))
y = (Dense(1, activation='relu'))(input_scalers)

model = Concatenate()([x, y])

x = Dense(64)(model)
x = Activation('relu')(x)

x = Dense(30)(x)
outputs = Activation('sigmoid')(x)

model = tf.keras.Model([inputs,input_scalers], outputs)

model.compile(loss='mse',optimizer='adam')

#make a diagram of the model structure
tf.keras.utils.plot_model(model, "my_first_model.png")

#performs the fitting
history = model.fit(
      x = [opacities,parameters],
      y = temperature,
      batch_size = batch_size,
      epochs = 100, #next time try 1000
      validation_split = 0.1,
      verbose = 2)

#saves model
model.save('my_model')

#plots loss and log(loss)
os.chdir('/localstorage/s1842521/plots')

epochs = range(len(history.history['loss']))

# Plot training and validation loss per epoch
plt.figure()
plt.plot(epochs, history.history['loss'],label='loss', color = 'green')
plt.plot(epochs, history.history['val_loss'],label='val_loss', color = 'mediumvioletred')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('lossplot.png',format='png')
plt.close()

plt.figure()
plt.plot(epochs, np.log10(history.history['loss']),label='loss', color = 'green')
plt.plot(epochs, np.log10(history.history['val_loss']),label='val_loss', color = 'mediumvioletred')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loglossplot.png',format='png')
plt.close()
