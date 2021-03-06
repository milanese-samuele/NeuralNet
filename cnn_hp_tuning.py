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

import random
import csv

def model_builder(hp, out_channels):
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

    # hyperparameter combinations
    hp = []

    for cfn in conv_filter_number:
        for cks in conv_kernel_size:
            for ps in pool_size:
                for dr in dropout_rates:
                    for dls in dense_layer_sizes:
                        for lr in learning_rates:
                            for lf in loss_functions:
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
    hyperparameter_tuning = True # false = training
    use_general_dataset = True # set to false for single patient dataset
    random_search = False # false = exhaustive gridsearch
    K = 10 # number of crossvalidation folds
    random_search_trials = 2
    csv_tuning_filename = 'hyperparameter_metrics.csv'

    patient_objects, labelset = select_patients(utils.pns, 5) #all patients with at least 5 classes
    general_batch, labelset = gen_batch(patient_objects, labelset, 100, ds=0.8) #all classes with at least 100 samples
    print(labelset)
    print([p.number for p in patient_objects])
    print(len(general_batch))
    output_size = len(labelset)
    general_patient_data, general_patient_data_labels, single_patient_data, single_patient_data_labels = next(generate_training_batches(patient_objects, general_batch, labelset))

    if use_general_dataset:
        inputs = general_patient_data
        labels = general_patient_data_labels
    else:
        inputs = single_patient_data
        labels = single_patient_data_labels

    if (hyperparameter_tuning):
        # hyperparameter gridsearch set-up
        hp_grid = hyperparameter_grid_builder()
        if random_search:
            random.shuffle(hp_grid)
            hp_grid = hp_grid[:random_search_trials]

    else:
        # Set desired architecture
        hp_grid = [[32, 7, 3, 0.5, 75, 0.1, 'categorical_crossentropy']] #best of tenfold gridsearch

    # Initialize model (average) metrics and hp containers
    models_metrics = []

    # save metrics to file
    with open(csv_tuning_filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        header = ['num_filters', 'kernel_size', 'pool_size', 'dropout_rates', 'dense_layer_size', 'learning_rate', 'loss_function', 'loss', 'acc', 'acc_std', 'tp', 'fp', 'tn', 'fn']
        writer.writerow(header)

        # Hyperparameter search loop
        for itr, hp in enumerate(hp_grid):
            # Print progress
            print(f'Model {itr + 1}/{len(hp_grid)}')

            # Initizalize per-fold score lists
            acc_per_fold = []
            loss_per_fold = []
            tp_per_fold = []
            fp_per_fold = []
            tn_per_fold = []
            fn_per_fold = []

            # Define the K-fold Cross Validator
            kfold = KFold(n_splits=K, shuffle=True)

            # K-fold Cross Validation model evaluation
            fold_no = 1
            for train, test in kfold.split(inputs, labels):
                #build model
                model = model_builder(hp, output_size)

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

            average_loss = np.mean(loss_per_fold)
            average_accuracy = np.mean(acc_per_fold)
            average_accuracy_std = np.std(acc_per_fold)
            average_tp = np.mean(tp_per_fold)
            average_fp = np.mean(fp_per_fold)
            average_tn = np.mean(tn_per_fold)
            average_fn = np.mean(fn_per_fold)

            models_metrics.append([average_loss, average_accuracy, average_accuracy_std, average_tp, average_fp, average_tn, average_fn])

            # write the data
            data = hp_grid[itr] + models_metrics[itr]
            writer.writerow(data)

        print('------------------------------------------------------------------------')
        print('BEST MODEL:')
        models_average_accuracy = np.array(models_metrics)[:,1]
        index_best_model = np.argmax(models_average_accuracy)
        print(f'> Average loss: {models_metrics[index_best_model][0]}')
        print(f'> Average accuracy: {models_metrics[index_best_model][1]} (+- {models_metrics[index_best_model][2]})')
        print(f'> Average tp rate: {models_metrics[index_best_model][3]}')
        print(f'> Average fp rate: {models_metrics[index_best_model][4]}')
        print(f'> Average tn rate: {models_metrics[index_best_model][5]}')
        print(f'> Average fn rate: {models_metrics[index_best_model][6]}')
        print(f'> Hyperparamers: {hp_grid[index_best_model]}')
        print('------------------------------------------------------------------------')

if __name__ == "__main__":
    main()
