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


def model_builder(hp, out_channels):
    #cfn1, cks1, ps1, cfn2, cks2, ps2, dls, dr = hp
    num_filters, kernel_size, pool_size, dropout_rates, dense_layer_size, learning_rate, loss_function = hp

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
    model.add(Dense(out_channels, activation='softmax'))

    # Set evaluation metrics
    metrics = set_metrics()

    model.compile(loss=loss_function, optimizer=tf.keras.optimizers.SGD(lr=learning_rate), metrics=metrics)

    #model.summary()

    return model

def hyperparameter_grid_builder():
    conv_filter_number = [8, 16, 32]
    conv_kernel_size = [3, 5, 7]
    pool_size = [3, 5]
    dense_layer_sizes = [25, 50, 75]
    dropout_rates = [0.3, 0.5, 0.7]
    learning_rates = [0.001, 0.01, 0.1]
    loss_functions = ['categorical_crossentropy', 'mean_squared_error']
    #downsampling_rates = [0.0, 0.5, 1.0]

    # hyperparameter combinations
    hp = []

    for cfn in conv_filter_number:
        for cks in conv_kernel_size:
            for ps in pool_size:
                for dr in dropout_rates:
                    for dls in dense_layer_sizes:
                        for lr in learning_rates:
                            for lf in loss_functions:
                                #for ds in downsampling_rates:
                                hp.append([cfn, cks, ps, dr, dls, lr, lf])

    print(f'Number of hyperparameter combinations: {len(hp)}')

    return hp

def set_metrics():
    return [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalseNegatives(name='fn')]

def main():
    # Set to true if you wish to tune hyperparameter using exhaustive grid search
    hyperparameter_tuning= True
    use_general_dataset=True

    if use_general_dataset:
        patient_data, _ = gen_tuning_batch(utils.pns, 5, 100, 0.5)
        labels = [w.btype for w in patient_data]
        labelset = list(set(labels))
        labels = np.asarray(utils.annotations_to_signal(labels, labelset))
        inputs = np.asarray([np.asarray(w.signal) for w in patient_data])
        # Reshape to fit model
        inputs = inputs.reshape(len(inputs), 114, 1)
        out_channels = len(labelset)

    else:
        # Inputs and labels from a preprocessed patient
        patient_data = balance_patient(208, 0.1, 3)
        labels = [w.btype for w in patient_data]
        # one hot encoding
        labels = np.asarray(utils.annotations_to_signal(labels, ["F", "V", "N"]))
        inputs = np.asarray([np.asarray(w.signal) for w in patient_data])
        # Reshape to fit model
        inputs = inputs.reshape(len(inputs), 114, 1)
        out_channels = 3

    if (hyperparameter_tuning):
        # hyperparameter gridsearch set-up
        hp_grid = hyperparameter_grid_builder()

    else:
        # Set desired architecture
        hp_grid = [[32, 5, 3, 0.3, 50, 0.1]]
        # best of ten fold with catcross, sgd, 3 epoch, batch 32 -> [8, 5, 3, 0.3, 25, 0.1]


    # Initialize model (average) accuracy, loss, and hp containers
    models = []
    models_average_accuracy = []
    models_average_accuracy_std = []
    models_average_loss = []
    models_average_tp = []
    models_average_fp = []
    models_average_tn = []
    models_average_fn = []

    # Hyperparameter gridsearch loop
    for itr, hp in enumerate(hp_grid):

        print(f'Model {itr + 1}/{len(hp_grid)}')

        # Initizalize per-fold score lists
        acc_per_fold = []
        loss_per_fold = []
        tp_per_fold = []
        fp_per_fold = []
        tn_per_fold = []
        fn_per_fold = []

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=10, shuffle=True)

        # K-fold Cross Validation model evaluation
        fold_no = 1
        for train, test in kfold.split(inputs, labels):
            #build model
            model = model_builder(hp, out_channels)

            model.fit(inputs[train], labels[train], epochs=3, batch_size=32, verbose=0) #tune batch size and epochs


            scores = model.evaluate(inputs[test],
                                    labels[test],
                                    batch_size=32,
                                    verbose=0)
            #print(model.metrics_names)
            #print(scores)
            loss_per_fold.append(scores[0])
            acc_per_fold.append(scores[1] * 100)
            tp_per_fold.append(scores[2])
            fp_per_fold.append(scores[3])
            tn_per_fold.append(scores[4])
            fn_per_fold.append(scores[5])


        models_average_accuracy.append(np.mean(acc_per_fold))
        models_average_accuracy_std.append(np.std(acc_per_fold))
        models_average_loss.append(np.mean(loss_per_fold))
        models_average_tp.append(np.mean(tp_per_fold))
        models_average_fp.append(np.mean(fp_per_fold))
        models_average_tn.append(np.mean(tn_per_fold))
        models_average_fn.append(np.mean(fn_per_fold))


    print('------------------------------------------------------------------------')
    print('BEST MODEL:')
    index_best_model = models_average_accuracy.index(max(models_average_accuracy))
    print(f'> Average accuracy: {models_average_accuracy[index_best_model]} (+- {models_average_accuracy_std[index_best_model]})')
    print(f'> Average loss: {models_average_loss[index_best_model]}')
    print(f'> Average tp rate: {models_average_tp[index_best_model]}')
    print(f'> Average fp rate: {models_average_fp[index_best_model]}')
    print(f'> Average tn rate: {models_average_tn[index_best_model]}')
    print(f'> Average fn rate: {models_average_fn[index_best_model]}')
    print(f'> Hyperparamers: {hp_grid[index_best_model]}')
    print('------------------------------------------------------------------------')


if __name__ == "__main__":
    main()
