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

import itertools


def model_builder(hp):
    #cfn1, cks1, ps1, cfn2, cks2, ps2, dls, dr = hp
    num_filters, kernel_size, pool_size, dropout_rates, dense_layer_size, learning_rate = hp

    model = Sequential()
    #Layer 1
    model.add(Conv1D(filters=num_filters, kernel_size=(kernel_size,), activation="relu", input_shape=(114, 1))) #tune filter and kernal size
    #Layer 2
    model.add(AveragePooling1D(pool_size=pool_size)) #tune pool size
    #Layer 3
    model.add(Flatten())
    #Layer 4
    model.add(Dropout(dropout_rates)) #tune dropout rate
    #Layer 5
    model.add(Dense(units=dense_layer_size, activation='relu')) #tune number of hidden units
    #Layer 6
    model.add(Dense(3, activation='softmax'))

    metrics = [
                tf.keras.metrics.Accuracy(name='accuracy'),
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalseNegatives(name='fn'),
                tf.keras.metrics.Precision(name='precision'),
             ]

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=learning_rate), metrics=['accuracy'])

    #model.summary()

    return model

def hyperparameter_grid_builder():
    conv_filter_number = [8, 16, 32] 
    conv_kernel_size = [3, 5, 7] 
    pool_size = [3, 5]
    dense_layer_sizes = [25, 50, 75] 
    dropout_rates = [0.3, 0.5, 0.7]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]

    # hyperparameter combinations
    hp = [] 

    for cfn in conv_filter_number:
        for cks in conv_kernel_size:
            for ps in pool_size:
                for dr in dropout_rates:
                    for dls in dense_layer_sizes:
                        for lr in learning_rates:
                            hp.append([cfn, cks, ps, dr, dls, lr])
    
    print(f'Number of hyperparameter combinations: {len(hp)}')

    return hp

def main():
    # Inputs and labels from a preprocessed patient
    patient_data = balance_patient(208, 0.1, 3)
    labels = [w.btype for w in patient_data]
    # one hot encoding
    labels = np.asarray(utils.annotations_to_signal(labels, ["F", "V", "N"]))
    inputs = np.asarray([np.asarray(w.signal) for w in patient_data])
    

    # Reshape to fit model 
    inputs = inputs.reshape(len(inputs), 114, 1)

    # hyperparameter gridsearch set-up
    hp_grid = hyperparameter_grid_builder()

    # Define model (average) accuracy, loss, and hp containers
    models = []
    models_average_accuracy = []
    models_average_accuracy_std = []
    models_average_loss = []

    # Hyperparameter gridsearch loop
    for itr, hp in enumerate(hp_grid):

        print(f'Model {itr + 1}/{len(hp_grid)}')


        # Define per-fold score lists
        acc_per_fold = []
        loss_per_fold = []

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=2, shuffle=True)

        # K-fold Cross Validation model evaluation
        fold_no = 1
        for train, test in kfold.split(inputs, labels):
            #build model
            model = model_builder(hp)

            model.fit(inputs[train], labels[train], epochs=3, verbose=0) #tune batch size and epochs
        

            scores = model.evaluate(inputs[test],
                                    labels[test],
                                    verbose=0)
            # print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])

        '''

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
        '''
    

        models_average_accuracy.append(np.mean(acc_per_fold))
        models_average_accuracy_std.append(np.std(acc_per_fold))
        models_average_loss.append(np.mean(loss_per_fold))
        models.append(model)

    
    print('------------------------------------------------------------------------')
    print('BEST MODEL:')
    index_best_model = models_average_accuracy.index(max(models_average_accuracy))
    print(f'> Average accuracy: {models_average_accuracy[index_best_model]} (+- {models_average_accuracy_std[index_best_model]})')
    print(f'> Average loss: {models_average_loss[index_best_model]}')
    print(models[index_best_model].summary())
    print(f'> Hyperparamers: {hp_grid[index_best_model]}')
    print('------------------------------------------------------------------------') 


if __name__ == "__main__":
    main()
