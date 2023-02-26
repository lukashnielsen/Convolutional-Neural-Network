import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, Concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import save_model, load_model
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn import preprocessing
import gc

gc.enable()

batch_size = 512

#load in processed data
temperature = np.load('processed_train_temperatures.npy')
pressure = np.load('pressure.npy')
parameters = np.load('processed_train_parameters.npy')
opacities = np.load('processed_train_opacities.npy')
temperatures_unchanged = np.load('temperature.npy')

#load model
model = load_model('my_model')

#perform additional training
history = model.fit(
      x = [opacities,parameters],
      y = temperature,
      batch_size = batch_size,
      epochs = 100, #next time try 1000
      validation_split = 0.1,
      verbose = 2)

#save model
model.save('my_model_100')

os.chdir('/localstorage/s1842521/plots')

# Get number of epochs
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

#predict values for training data
predictedtemps = model.predict(x=[opacities,parameters],
                       batch_size=512,)


#unnormalize predicted data
#array1 and array2 have to be the same shape
#array1 is the original array and array 2 is the array which has to be unnormalized wrt array 1
def unnormalize(array1,array2):
    temp_array=np.transpose(array1)
    array3=np.zeros(temp_array.shape)
    array2 = np.transpose(array2)
    for i in range(30):
        maxtemp=temp_array[i].max()
        mintemp=temp_array[i].min()
        array3[i] = array2[i]*(maxtemp-mintemp)+mintemp
    array3 = np.transpose(array3)
    return array3

#predict is the unnormalized predicted data
predict=unnormalize(temperatures_unchanged[0:75000],predictedtemps)

#change directory to plot 100 images of predicted temperature vs real temperature
os.chdir('/localstorage/s1842521/plots/train/')
for i in range(100):
    plt.figure()
    plt.plot(predict[i],np.log10(pressure[i]),color = 'green',label='predicted temp')
    plt.plot(temperatures_unchanged[i],np.log10(pressure[i]),color = 'mediumvioletred',label='real temp')
    plt.title('Predicted vs Real Temperature')
    plt.xlabel('temperature (K)')
    plt.ylabel('log10(pressure)')
    plt.legend()
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig('trainplot'+str(i)+'.png',format='png')
    plt.close()

#plots validation data
#validation data is sectioned off the end of the model, so by this model choosing val_split = 0.1, the validation data is the last 10% of training data
os.chdir('/localstorage/s1842521/plots/validation/')
for i in range(100):
    plt.figure()
    plt.plot(predict[i+67500],np.log10(pressure[i+67500]),color = 'green',label='predicted temp')
    plt.plot(temperatures_unchanged[i+67500],np.log10(pressure[i+67500]),color = 'mediumvioletred',label='real temp')
    plt.title('Predicted vs Real Temperature')
    plt.xlabel('temperature (K)')
    plt.ylabel('log10(pressure)')
    plt.legend()
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig('valplot'+str(i)+'.png',format='png')
    plt.close()

#delete old data to save ram
del opacities
del temperature
del parameters

os.chdir('/localstorage/s1842521/')

#load test set
opacities = np.load('processed_test_opacities.npy')
temperature = np.load('processed_test_temperatures.npy')
parameters = np.load('processed_test_parameters.npy')

#once again, predict values but this time they are unseen
predictedtest = model.predict(x=[opacities,parameters],
                       batch_size=512,)

predicttest = unnormalize(temperatures_unchanged[75000:100000],predictedtest)

os.chdir('/localstorage/s1842521/plots/test/')
for i in range(100):
    plt.figure()
    plt.plot(predicttest[i],np.log10(pressure[i+75000]),color = 'green',label='predicted temp')
    plt.plot(temperatures_unchanged[i+75000],np.log10(pressure[i+75000]),color = 'mediumvioletred',label='real temp')
    plt.title('Predicted vs Real Temperature')
    plt.xlabel('temperature (K)')
    plt.ylabel('log10(pressure)')
    plt.legend()
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig('testplot'+str(i)+'.png',format='png')
    plt.close()
