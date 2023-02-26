# Convolutional-Neural-Network
A model to emulate the radiative transfer of exoplanet atmospheres.
I built a model which trained on images of the opacity of the atmosphere at different pressures with different wavelength bins, along with parameters that are the distance to the star, internal temperature of the planet and temperature of the star to predict the temperature structure of the atmosphere

process.py - the code which performs the necessary preprocessing to the data, which involves selecting the first three parameter values of 8 and discarding the other 5, combining the average opacity and scattering opacity together into one array and normalising the code with respect to its columns, as it has to be normalised with respect to the pressure values
master.py - the code which builds the model and predicts the loss and log(loss) values of the code
temp.py - performs additional training without having to build the model again, and predicts new temperature values based on the training opacities/parameters and everntually the test opacities/parameters
