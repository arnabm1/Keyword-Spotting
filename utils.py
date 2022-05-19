import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger

def lr_schedule(epoch):
    epochs = 100
    lr = 1e-3
    if epoch > epochs*180//200:
        lr *= 0.5e-3
    elif epoch > epochs*160//200:
        lr *= 1e-3
    elif epoch > epochs*120//200:
        lr *= 1e-2
    elif epoch > epochs*80//200:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def load_data(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Training sets loaded!")
    return X, y

def prepare_dataset(data_path, validation_size):
    """Creates train, validation and test sets.
    :param data_path (str): Path to json file containing data
    :param test_size (flaot): Percentage of dataset used for testing
    :param validation_size (float): Percentage of train set used for cross-validation
    :return X_train (ndarray): Inputs for the train set
    :return y_train (ndarray): Targets for the train set
    :return X_validation (ndarray): Inputs for the validation set
    :return y_validation (ndarray): Targets for the validation set
    :return X_test (ndarray): Inputs for the test set
    :return X_test (ndarray): Targets for the test set
    """

    # load dataset
    X, y = load_data(data_path)

    # create train, validation, test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size)

    # add an axis to nd array
    X_train = X_train[..., np.newaxis]
    # X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation

def onehot(labels):
  enc = OneHotEncoder()
  enc.fit(np.unique(labels).reshape(-1, 1))
  labels = enc.transform(labels.reshape(-1, 1)).toarray()
  num_classes = labels.shape[1]

  return labels
