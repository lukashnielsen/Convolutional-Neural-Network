import numpy as np
import sklearn
from sklearn import preprocessing
#
temperatures = np.load('temperature.npy') #24 Mb
pressure = np.load('pressure.npy') #24 Mb
opacities_scat = np.load('opacity_scat.npy') #20.7 Gb
parameters = np.load('parameters.npy') #6.4 Mb
opacities = np.load('opacity_aver.npy') #20.7 Gb

#parameters is a (100000,8) however we only need the first 3 in each row,
#so this selects the first 3
def change_params(a,c):
  b = np.zeros(c)
  for i in range((len(a))):
    b[i][0] = a[i][0]
    b[i][1] = a[i][1]
    b[i][2] = a[i][2]
  return b

#redefines parameters
parameters = change_params(parameters,(100000,3))

#normalise temps and parameters based on columns
min_max_scaler = preprocessing.MinMaxScaler()
temperature = min_max_scaler.fit_transform(temperatures)
parameters = min_max_scaler.fit_transform(parameters)

#normalise average opacities and scattering opacities based on columns
opacities = opacities.reshape(3000000,862)
opacities = np.log10(opacities)
opacities = min_max_scaler.fit_transform(opacities)
opacities = opacities.reshape(100000,30,862)

opacities_scat = opacities_scat.reshape(3000000,862)
opacities_scat = np.log10(opacities_scat)
opacities_scat = min_max_scaler.fit_transform(opacities_scat)
opacities_scat = opacities_scat.reshape(100000,30,862)

opacities = np.stack((opacities,opacities_scat), axis = -1)

np.save('processed_train_opacities.npy',opacities[0:75000])
np.save('processed_train_parameters.npy',parameters[0:75000])
np.save('processed_train_temperatures.npy',temperature[0:75000])

np.save('processed_test_opacities.npy',opacities[75000:100000])
np.save('processed_test_parameters.npy',parameters[75000:100000])
np.save('processed_test_temperatures.npy',temperature[75000:100000])
