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


def set_metrics():
    return [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalseNegatives(name='fn')]

def main():
    use_general_dataset = True # set to false for single patient dataset

    if use_general_dataset:
        patient_names, labelset = select_patients(utils.pns, 5)
        general_batch, labelset = gen_tuning_batch(patient_names,
                                                   labelset, 100, 0.3)
        print(labelset)
        print([p.number for p in patient_names])
        print(len(general_batch))
        generate_training_batches(patient_names, general_batch, labelset)

        # make one patient data
        patient_data = balance_patient(specific_patient.number)
        patient_labels = [w.btype for w in patient_data]
        print(f'Number of classes: {len(labelset)}')
        patient_labels = np.asarray(utils.annotations_to_signal(labels, labelset))
        patient_inputs = np.asarray([np.asarray(w.signal) for w in patient_data])
        # Reshape to fit model
        patient_inputs = inputs.reshape(len(inputs), 114, 1)

        general_data = gen_tuning_batch(patient_names, labelset, 500, )

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


    # Set desired architecture
    hp = [32, 7, 3, 0.5, 75, 0.1, 'categorical_crossentropy']

    # Initialize model (average) metrics and hp containers
    models_metrics = []

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

    average_loss = np.mean(loss_per_fold)
    average_accuracy = np.mean(acc_per_fold)
    average_accuracy_std = np.std(acc_per_fold)
    average_tp = np.mean(tp_per_fold)
    average_fp = np.mean(fp_per_fold)
    average_tn = np.mean(tn_per_fold)
    average_fn = np.mean(fn_per_fold)

    models_metrics.append([average_loss, average_accuracy, average_accuracy_std, average_tp, average_fp, average_tn, average_fn])


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
    print(f'> Hyperparamers: {hp}')
    print('------------------------------------------------------------------------')

if __name__ == "__main__":
    main()
