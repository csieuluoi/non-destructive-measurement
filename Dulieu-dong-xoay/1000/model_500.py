from tensorflow.keras.layers import Conv2D, Input, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
# import tensorflow_docs as tfdocs
# import tensorflow_docs.plots
# import tensorflow_docs.modeling

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import os
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import date, datetime

## define some constants
CONFIG = {
    'LR': 1e-3,
    'EPOCHS': 20000,
    'BATCH_SIZE': 256
}


class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = Conv2D(kernel_size = (3, 3), filters = 64, strides = 2, activation = 'relu')
        self.maxpool1 = MaxPooling2D((2, 2), strides = 1)

        self.conv2 = Conv2D(kernel_size = (3, 3), filters = 64, strides = 2, activation = 'relu')
        self.maxpool2 = MaxPooling2D((2, 2), strides = 1)

        self.conv3 = Conv2D(kernel_size = (3, 3), filters = 128, strides = 1, activation = 'relu')
        self.maxpool3 = MaxPooling2D((2, 2), strides = 1)

        self.conv4 = Conv2D(kernel_size = (3, 3), filters = 128, strides = 1, activation = 'relu')
        self.maxpool4 = MaxPooling2D((2, 2), strides = 1)

        self.flatten = Flatten()
        self.dense1 = Dense(256, activation = 'relu')
        self.dropout = Dropout(0.3)
        self.dense2 = Dense(1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        # if training:
        x = self.dropout(x)
        return self.dense2(x)





def build_model():
    input = Input(shape = (40, 40, 1))
    x = Conv2D(kernel_size = (3, 3), filters = 64, strides = 2, activation = 'relu')(input)
    x = MaxPooling2D((2, 2), strides = 1)(x)

    x= Conv2D(kernel_size = (3, 3), filters = 64, strides = 2, activation = 'relu')(x)
    x = MaxPooling2D((2, 2), strides = 1)(x)

    x = Conv2D(kernel_size = (3, 3), filters = 128, strides = 1, activation = 'relu')(x)
    x = MaxPooling2D((2, 2), strides = 1)(x)

    x = Conv2D(kernel_size = (3, 3), filters = 128, strides = 1, activation = 'relu')(x)
    x = MaxPooling2D((2, 2), strides = 1)(x)

    x = Flatten()(x)
    x = Dense(256, activation = 'relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(1)(x)

    model = Model(inputs = [input], outputs = [output])
    # optimizer = Adam(learning_rate = 1e-3)

    # model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', 'mse'])

    return model

def get_call_back(config):
    # day = date.today()
    # day_str = day.strftime("%Y-%m-%d")

    date_time = datetime.now()
    date_time_str = date_time.strftime("%Y-%m-%d_%H-%M-%S")
    # dd/mm/YY
    outputFolder = f"./checkpoint/{date_time_str}-[{config['EPOCHS']}-{config['BATCH_SIZE']}-{config['LR']}]"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    filepath=outputFolder+"/model-{epoch:03d}-{loss:5.2f}-{val_loss:5.2f}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=False,
                                 save_weights_only=True,
                                 period=1000)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=1, mode='min')

    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                   factor=0.1,
                                   cooldown=0,
                                   patience=500,
                                   min_lr=1e-4)
    log = tf.keras.callbacks.TensorBoard(log_dir=f"./log")
    return [lr_reducer, early_stopping, checkpoint, log]


def load_checkpoint(checkpoint_dir = './checkpoint/2020-09-13_00-41-11-[20000-256-0.001]/model-9000-4922.68.hdf5'):
    model = MyModel()
    input = Input(shape = (20, 20, 1))

    model.call(inputs = input)

    model.load_weights(checkpoint_dir)

    return model

def get_data():
    # Loading data
    img_size = (40, 40)
    step = 15
    # X = np.load(f'train_data/{step}_train_cut_imgs.npy')
    # Y = np.load(f'train_data/{step}_labels_cut.npy')

    X = np.load(f'train_data/{step}_{img_size}_train_cut_imgs.npy')
    Y = np.load(f'train_data/{step}_{img_size}_labels_cut.npy')

    X = np.expand_dims(X, axis = 3)/255
    # X = (255 - X)/255
    # add noise to label
    # np.random.seed(10)
    mu, sigma = 0, 5 # mean and standard deviation
    noise = np.random.normal(mu, sigma, Y.shape[0])
    Y_noise = Y + noise
    print(Y_noise)
    # X_train, X_test = X[152:,:,:,:], X[:152,:,:,:]
    # y_train, y_test = Y_noise[152:], Y_noise[:152]
    # X_train, X_test, y_train, y_test = train_test_split(X, Y_noise, test_size = 0.2, shuffle = True, random_state = 29)


    return X, Y_noise

def train_kfold(num_folds = 8):
    X, Y  = get_data()

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=False)

    mae_per_fold = []
    mse_per_fold = []

    fold_no = 1
    histories = {}
    for train, test in kfold.split(X, Y):
        model = build_model()
        call_back_list = get_call_back(CONFIG)[:2]

        optimizer  = Adam(learning_rate = CONFIG['LR'])
        model.compile(loss = 'mean_squared_error', optimizer = optimizer, metrics = ['mae', 'mse'])
        history = model.fit(X[train], Y[train],
            epochs = CONFIG['EPOCHS'],
            batch_size = CONFIG['BATCH_SIZE'],
            callbacks = call_back_list,
            validation_data = (X[test], Y[test]),
            shuffle = True,
            verbose = 1)

        histories[f'history_{fold_no}'] = history.history
        # Generate generalization metrics
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}')
        mae_per_fold.append(scores[1])
        mse_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1

    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(mae_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - MSE: {mse_per_fold[i]} - MAE: {mae_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> MAE: {np.mean(mae_per_fold)} (+- {np.std(mae_per_fold)})')
    print(f'> MSE: {np.mean(mse_per_fold)}')
    print('------------------------------------------------------------------------')

    # print(histories)
    with open('histories.pkl', 'wb') as f:
        pickle.dump(histories, f)

if __name__ == '__main__':


    # estimator = KerasRegressor(build_fn=build_model, epochs=5000, batch_size=16, verbose=0)

    # kfold = KFold(n_splits=8, shuffle = True)
    # results = cross_val_score(estimator, X, Y_noise, cv=kfold)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    # print(Y)

    # model = MyModel()
    # input = Input(shape = X.shape[1:])

    # model.call(inputs = input)
    # lr = 1e-3
    # optimizer = Adam(learning_rate = lr)

    # model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', 'mse'])
    # call_back_list = get_call_back(CONFIG)
    # history = model.fit(X_train, y_train,
    #     epochs = CONFIG['EPOCHS'],
    #     batch_size = CONFIG['BATCH_SIZE'],
    #     validation_data = (X_test,y_test),
    #     callbacks=call_back_list,
    #     shuffle = True)

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()


    # # X_eval = np.load('29_train_cut_imgs.npy')/255
    # # Y_eval = np.load('29_labels_cut.npy')
    # # X_eval = (255 - X_eval)/255.

    # X_eval = np.load(f'train_data/29_{img_size}_train_cut_imgs.npy')/255
    # Y_eval = np.load(f'train_data/29_{img_size}_labels_cut.npy')
    # X_eval = np.expand_dims(X_eval, axis = 3)


    # # load checkpoint
    # # model = load_checkpoint(checkpoint_dir =  './checkpoint/2020-09-13_00-41-11-[20000-256-0.001]/model-9000-4922.68.hdf5')
    # # lr = 1e-3
    # # optimizer = Adam(learning_rate = lr)

    # # model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', 'mse'])
    # # history = model.fit(X_train, y_train,
    # #     epochs = EPOCHS,
    # #     batch_size = BATCH_SIZE,
    # #     validation_data = (X_test,y_test),
    # #     callbacks=call_back_list,
    # #     shuffle = True)
    # # plt.plot(history.history['loss'])
    # # plt.plot(history.history['val_loss'])
    # # plt.title('model loss')
    # # plt.ylabel('loss')
    # # plt.xlabel('epoch')
    # # plt.legend(['train', 'val'], loc='upper left')
    # # plt.show()
    # Y_pred = model.predict(X_eval)
    # loss = model.evaluate(X_eval, Y_eval, verbose=2)

    # print(loss)
    # print(Y_pred)
    # print(Y_eval)
    # print(Y_eval.shape)

    train_kfold(num_folds = 8)
