import tensorflow as tf
from tensorflow.python.keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config=config)
K.set_session(sess) 

from sklearn.model_selection import KFold
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, AveragePooling1D, Dropout
import matplotlib.pyplot as plt

from aug import *
import utils


def model_builder():
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=13, activation="relu", input_shape=(114, 1))) #tune filter and kernal size
    model.add(AveragePooling1D(pool_size=3)) #tune pool size
    model.add(Conv1D(filters=15, kernel_size=32, activation="relu")) #tune filter and kernal size
    model.add(AveragePooling1D(pool_size=3)) #tune pool size
    model.add(Flatten())
    model.add(Dropout(0.5)) #tune dropout rate
    model.add(Dense(100, activation='relu')) #tune number of hidden units
    model.add(Dense(3, activation='softmax'))

    metrics = [
            tf.keras.metrics.Accuracy(name='accuracy'),
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.Precision(name='precision'),
             ]

    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    model.summary()

    return model


def main():
    # Inputs and labels from a preprocessed patient
    patient_data = balance_patient(208, 0.1, 3)
    labels = [w.btype for w in patient_data]
    # one hot encoding
    labels = np.asarray(utils.annotations_to_signal(labels, ["F", "V", "N"]))
    inputs = np.asarray([np.asarray(w.signal) for w in patient_data])

    # Define per-fold score lists
    acc_per_fold = []
    loss_per_fold = []


    # reshape to fit model (1 for grayscale)
    inputs = inputs.reshape(len(inputs), 114, 1)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=2, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, labels):
        #build model
        model = model_builder()

        model.fit(inputs[train], labels[train], epochs=3) 
    

        scores = model.evaluate(inputs[test],
                                labels[test])
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------') 



if __name__ == "__main__":
    main()
