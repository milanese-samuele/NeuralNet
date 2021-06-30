from cnn import build_model
import utils
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, AveragePooling1D, Dropout
from tensorflow.python.keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

from sklearn.model_selection import KFold
import numpy as np

from aug import *


def model_builder(input_shape, n_outputs):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=13, activation="relu", input_shape=input_shape)) #tune filter and kernal size
    model.add(AveragePooling1D(pool_size=3)) #tune pool size
    model.add(Conv1D(filters=16, kernel_size=13, activation="relu")) #tune filter and kernal size
    model.add(AveragePooling1D(pool_size=3)) #tune pool size
    model.add(Dropout(0.5)) #tune dropout rate
    model.add(Dense(100, activation='relu')) #tune number of hidden units
    model.add(Dense(n_outputs, activation='softmax'))

    metrics = [
            tf.keras.metrics.Accuracy(name='accuracy'),
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.Precision(name='precision'),
             ]

    model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),# tune learning rate and optimizer type
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(), # tune loss function type
                    metrics=metrics
                    )

    return model


def main():

    # Inputs and labels from a preprocessed patient
    patient_data = balance_patient(208, 0.1, 3)
    labels = [w.btype for w in patient_data]
    # one hot encoding
    labels = utils.annotations_to_signal(labels, ["F", "V", "N"])
    inputs = np.asarray([np.asarray(w.signal) for w in patient_data])

    '''
    print(inputs[0])
    print(type(inputs))'''

    # Size of a single heartbeat
    input_shape = (len(inputs[0]), 1)

    # Define per-fold score lists
    acc_per_fold = []
    loss_per_fold = []

    # define K
    num_folds = 3

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, labels):
        '''
        model = model_builder(input_shape, 3)
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        history = model.fit([inputs [i] for i in train],
                            [labels [i] for i in train],
                            epochs=3,
                            batch_size=32) #tune batch size

        scores = model.evaluate([inputs [i] for i in test],
                                [labels [i] for i in test],
                                verbose=0)
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
    print('------------------------------------------------------------------------') '''



if __name__ == "__main__":
    main()
