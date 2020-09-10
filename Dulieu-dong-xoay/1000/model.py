from tensorflow.keras.layers import Conv2D, Input, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# import tensorflow_docs as tfdocs
# import tensorflow_docs.plots
# import tensorflow_docs.modeling

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np

def build_model(lr = 1e-3):
    input = Input(shape = (24, 20, 1))
    x = Conv2D(kernel_size = (3, 3), filters = 16, strides = 2, activation = 'relu')(input)
    x = MaxPooling2D((2, 2), strides = 1)(x)
    x = Conv2D(kernel_size = (3, 3), filters = 8, strides = 1, activation = 'relu')(x)
    x = MaxPooling2D((2, 2), strides = 1)(x)
    x = Flatten()(x)
    x = Dense(32, activation = 'relu')(x)
    output = Dense(1)(x)
    optimizer = Adam(learning_rate = lr)
    model = Model(inputs = [input], outputs = [output])
    # model.summary()
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', 'mse'])

    return model


if __name__ == '__main__':
    model = build_model()
    X = np.load('train_imgs.npy')
    X = np.expand_dims(X, axis = 3)
    # X = X
    print(X.shape)
    Y = np.load('labels.npy')
    Y = Y/10
    print(Y)
    # add noise to label
    np.random.seed(1)
    mu, sigma = 0, 1 # mean and standard deviation
    noise = np.random.normal(mu, sigma, Y.shape[0])
    Y_noise = Y + noise
    # print(Y_noise)
    # estimator = KerasRegressor(build_fn=build_model, epochs=50, batch_size=16, verbose=0)

    # kfold = KFold(n_splits=4)
    # results = cross_val_score(estimator, X, Y, cv=kfold)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    print(Y)
    history = model.fit(X, Y, epochs = 200, batch_size = 16, validation_split = 0.2)



