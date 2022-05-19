import json
import numpy as np
import tensorflow as tf
import utils
from keras.optimizers import *
from qkeras import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from utils import load_data, lr_schedule, prepare_dataset, onehot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, Activation, Input
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import InputLayer, Reshape, Flatten, Dense, Dropout

DATA_PATH = "data.json"
BATCH_SIZE = 32

X_train, y_train, X_validation, y_validation = prepare_dataset(DATA_PATH, validation_size=0.2)

print(X_train.shape)
print(y_train.shape)
print(X_validation.shape)
print(y_validation.shape)

scale = np.array([4,2,1,0.5])
quant = np.array([8,4,2])

for s in (scale):
	for q in (quant):
		for run in range(2):

			epochs = 2

			print('\n ######_' + 'Start execution_' + repr(run+1) + '_for ks_' + '1sec_w_' + repr(s) + 's_with_' + repr(q) + 'q_and_' + repr(epochs) + '_epochs_######')


			if __name__ == '__main__':

				batch_size = 32

				def CreateQModel(shape, nb_classes):

					x = x_in = Input(shape)
					x = QConv2D(int(64*s), (3,3), kernel_quantizer="quantized_bits("+repr(q)+",0,1)", bias_quantizer="quantized_bits("+repr(q)+")",
								kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
					x = QActivation("quantized_bits("+repr(q)+")")(x)
					# x = QBatchNormalization()(x)

					x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)

					x = QConv2D(int(32*s), (3,3), kernel_quantizer="quantized_bits("+repr(q)+",0,1)", bias_quantizer="quantized_bits("+repr(q)+")",
								kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
					x = QActivation("quantized_bits("+repr(q)+")")(x)
					# x = QBatchNormalization()(x)

					x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)

					x = QConv2D(int(32*s), (2,2), kernel_quantizer="quantized_bits("+repr(q)+",0,1)", bias_quantizer="quantized_bits("+repr(q)+")",
								kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
					x = QActivation("quantized_bits("+repr(q)+")")(x)
					# x = QBatchNormalization()(x)

					x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)

					x = Flatten()(x)

					x = QDense(int(64*s), kernel_quantizer="quantized_bits("+repr(q)+",0,1)", bias_quantizer="quantized_bits("+repr(q)+")")(x)
					x = QActivation("quantized_bits("+repr(q)+")")(x)
					x = Dropout(0.3)(x)

					x = QDense(nb_classes, kernel_quantizer="quantized_bits("+repr(q)+",0,1)", bias_quantizer="quantized_bits("+repr(q)+")")(x)
					x = Activation("softmax")(x)

					model = Model(inputs=x_in, outputs=x)
					model.summary()

					return model

				qmodel = CreateQModel(X_train.shape[1:], 30)

				qmodel.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])

				checkpoint = ModelCheckpoint('Trained_models/ks_' + repr(q) + '_bit_' + '1sec_w_' + repr(s) + 's_run_' + repr(run+1) + '.hdf5', 
											  verbose=2, save_best_only=True, monitor='val_accuracy', mode='max')

				lr_scheduler = LearningRateScheduler(lr_schedule)
				lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
				callbacks = [checkpoint, lr_reducer, lr_scheduler]

				qmodel.fit(
						X_train,
						y_train,
						steps_per_epoch = batch_size,
						epochs = epochs,
						verbose = 2,
						shuffle = True,
						validation_data = (X_validation, y_validation),
						validation_steps = batch_size,
						callbacks = callbacks)

				print('\n ######_' + 'End of execution_' + repr(run+1) + '_for ks_' + '1sec_w_' + repr(s) + 's_with_' + repr(q) + 'q_and_' + repr(epochs) + '_epochs_######')