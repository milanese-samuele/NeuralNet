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

def compile_model(model, loss_function, learning_rate):
    # Set evaluation metrics
    metrics = set_metrics()
    model.compile(loss=loss_function, optimizer=tf.keras.optimizers.SGD(lr=learning_rate), metrics=metrics)

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

    compile_model(model, loss_function, learning_rate)

    #model.summary()

    return model


def set_metrics():
    return [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalseNegatives(name='fn')]


def generate_trained_general_model(inputs, labels, hp, output_size):
    #build model
    model = model_builder(hp, output_size)
    model.fit(inputs, labels, epochs=3, batch_size=32, verbose=0) 

    print('General Model:')
    model.summary()

    return model


def k_fold_crossvalidation_training(inputs, labels, hp, output_size, model=None):
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
        if model == None: 
            model = model_builder(hp, out_channels)
        else:
            compile_model(model, loss_function=hp[-1], learning_rate=hp[-2])


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

    print('------------------------------------------------------------------------')
    print('MODEL:')
    tp = np.mean(tp_per_fold)
    fp = np.mean(fp_per_fold)
    tn = np.mean(tn_per_fold)
    fn = np.mean(fn_per_fold)
    print(f'> Average loss: {np.mean(loss_per_fold)}')
    print(f'> Average accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Average tp rate: {tp}')
    print(f'> Average fp rate: {fp}')
    print(f'> Average tn rate: {tn}')
    print(f'> Average fn rate: {fn}')
    print(f'> Average F1-score: {tp/(tp+0.5(fp+fn))}')
    print(f'> Hyperparamers: {hp}') 
    model.summary()
    print('------------------------------------------------------------------------')

def main():
    use_general_dataset = True # set to false for single patient dataset
    transfer_learning = True

    if use_general_dataset:
        patient_data, patient_codes = gen_tuning_batch(utils.pns, 5, 100, 0.8)
        labels = [w.btype for w in patient_data]
        labelset = list(set(labels))
        print(f'Number of classes: {len(labelset)}')
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

    

    # Set desired architecture
    hp = [32, 7, 3, 0.5, 75, 0.1, 'categorical_crossentropy']
    
    if transfer_learning:
        general_model = generate_trained_general_model(inputs, labels, hp, out_channels)
        for layer in general_model.layers[:5]:
            layer.trainable = False
        for layer in general_model.layers:
            print(layer, layer.trainable)
        k_fold_crossvalidation_training(inputs, labels, hp, out_channels, general_model)
    else:
        k_fold_crossvalidation_training(inputs, labels, hp, out_channels)
        

    

if __name__ == "__main__":
    main()
