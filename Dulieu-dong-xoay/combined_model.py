from tensorflow.keras.layers import Conv2D, Input, Dense, Flatten, MaxPooling2D, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from model_400 import Model400
from model_500 import Model500

from datetime import date, datetime

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import json
import pickle

CONFIG = {
	'LR': 1e-3,
	'EPOCHS': 20000,
	'BATCH_SIZE': 256
}

def get_data(step = 15, img_size = (40, 40)):

	X_500 = np.load(f'1000/train_data/{step}_{img_size}_train_cut_imgs.npy')/255
	Y_500 = np.load(f'1000/train_data/{step}_{img_size}_labels_cut.npy')
	X_500 = np.expand_dims(X_500, axis = 3)

	X_400 = np.load(f'1003/train_data/{step}_train_cut_imgs.npy')/255
	Y_400 = np.load(f'1003/train_data/{step}_labels_cut.npy')
	X_400 = np.expand_dims(X_400, axis = 3)

	X_eval_500 = np.load(f'1000/train_data/29_{img_size}_train_cut_imgs.npy')/255
	Y_eval_500 = np.load(f'1000/train_data/29_{img_size}_labels_cut.npy')
	X_eval_500 = np.expand_dims(X_eval_500, axis = 3)

	X_eval_400 = np.load(f'1003/train_data/29_train_cut_imgs.npy')/255
	Y_eval_400 = np.load(f'1003/train_data/29_labels_cut.npy')
	X_eval_400 = np.expand_dims(X_eval_400, axis = 3)

	print(Y_eval_400 - Y_eval_500)
	return (X_500, X_400, X_eval_500, X_eval_400, Y_400, Y_eval_400)

def build_model(load_weights = True, freeze = False):
	input_400 = Input(shape = (40, 40, 1))
	input_500 = Input(shape = (40, 40, 1))

	model_400 = Model400().call(input_400)
	model_500 = Model500().call(input_500)

	if load_weights:
		model_400.load_weights('model-400.hdf5')
		model_500.load_weights('model-500.hdf5')
	if freeze:
		for layer in model_400.layers[:-4]:
			layer.trainable = False
		for layer in model_500.layers[:-4]:
			layer.trainable = False

	flatten_400 = model_400.layers[-4].output
	flatten_500 = model_500.layers[-4].output

	concat = Concatenate()([flatten_400, flatten_500])
	dense1 = Dense(256, activation = 'relu')(concat)
	dropout1 =  Dropout(0.4)(dense1)
	out = Dense(1)(dropout1)

	combined_model = Model(inputs = [input_400, input_500], outputs = [out])

	# combined_model.summary()

	return combined_model

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
	log_dir = os.path.join('log')
	log = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
	return [lr_reducer, early_stopping, checkpoint, log]

def main():
	X_500, X_400, X_eval_500, X_eval_400, Y_400, Y_eval_400 = get_data()

	X_train_500, X_test_500, Y_train, Y_test = train_test_split(X_500, Y_400, test_size = 0.2, shuffle = True, random_state = 200)
	X_train_400, X_test_400, Y_train_, Y_test_ = train_test_split(X_400, Y_400, test_size = 0.2, shuffle = True, random_state = 200)

	# X_train_500, X_test_500 = X_500[152:], X_500[:152]
	# X_train_400, X_test_400 = X_400[152:], X_400[:152]
	# Y_train, Y_test = Y_400[152:], Y_400[:152]

	print(Y_train - Y_train_)
	print(X_400.shape, X_500.shape)
	model = build_model(load_weights = False, freeze = False)

	call_back_list = get_call_back(CONFIG)
	optimizer  = Adam(learning_rate = CONFIG['LR'])
	model.compile(loss = 'mean_squared_error', optimizer = optimizer, metrics = ['mae', 'mse'])

	history = model.fit([X_train_400, X_train_500], Y_train,
		epochs = CONFIG['EPOCHS'],
		batch_size = CONFIG['BATCH_SIZE'],
		validation_data = ([X_test_400, X_test_500], Y_test),
		callbacks = call_back_list,
		shuffle = True)


	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()
	model.evaluate([X_eval_400, X_eval_500], Y_eval_400)

def train_kfold(num_folds = 8):
	X_500, X_400, X_eval_500, X_eval_400, Y_400, Y_eval_400 = get_data()

	# Define the K-fold Cross Validator
	kfold = KFold(n_splits=num_folds, shuffle=False)

	mae_per_fold = []
	mse_per_fold = []

	fold_no = 1
	histories = {}
	for train, test in kfold.split(X_500, Y_400):
		model = build_model(load_weights = False, freeze = False)
		call_back_list = get_call_back(CONFIG)[:2]

		optimizer  = Adam(learning_rate = CONFIG['LR'])
		model.compile(loss = 'mean_squared_error', optimizer = optimizer, metrics = ['mae', 'mse'])
		history = model.fit([X_500[train], X_400[train]], Y_400[train],
			epochs = CONFIG['EPOCHS'],
			batch_size = CONFIG['BATCH_SIZE'],
			callbacks = call_back_list,
			validation_data = ([X_500[test], X_400[test]], Y_400[test]),
			shuffle = True,
			verbose = 1)

		histories[f'history_{fold_no}'] = history.history
		# Generate generalization metrics
		scores = model.evaluate([X_500[test], X_400[test]], Y_400[test], verbose=0)
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
	with open('histories/histories.pkl', 'wb') as f:
		pickle.dump(histories, f)

if __name__ == '__main__':
	# main()
	train_kfold(num_folds = 8)
