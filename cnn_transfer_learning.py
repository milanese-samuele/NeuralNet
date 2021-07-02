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

def plot_box_and_whisker(transfer_learning_acc, non_transfer_learning_acc):
    data = [transfer_learning_acc, non_transfer_learning_acc]

    fig1, ax1 = plt.subplots()
    ax1.set_title('10-fold Crossvalidation Accuracy: Transfer Learning vs Non-Transfer Learning')
    ax1.boxplot(data)

    plt.show()


def k_fold_crossvalidation_training(inputs, labels, hp, output_size, K, model=None, verbose=1):
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
    kfold = KFold(n_splits=K, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, labels):
        #build model
        if model == None: 
            model = model_builder(hp, output_size)
        else:
            compile_model(model, loss_function=hp[-1], learning_rate=hp[-2])


        history = model.fit(inputs[train], labels[train], epochs=3, batch_size=32, verbose=0) #tune batch size and epochs


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
    
    average_acc = np.mean(acc_per_fold)
    tp = np.mean(tp_per_fold)
    fp = np.mean(fp_per_fold)
    tn = np.mean(tn_per_fold)
    fn = np.mean(fn_per_fold)
    
    if verbose:
        print('------------------------------------------------------------------------')
        print('MODEL:')
        print(f'> Average loss: {np.mean(loss_per_fold)}')
        print(f'> Average accuracy: {average_acc} (+- {np.std(acc_per_fold)})')
        print(f'> Average tp rate: {tp}')
        print(f'> Average fp rate: {fp}')
        print(f'> Average tn rate: {tn}')
        print(f'> Average fn rate: {fn}')
        #print(f'> Average F1-score: {tp/(tp+0.5(fp+fn))}')
        print(f'> Hyperparamers: {hp}') 
        model.summary()
        print('------------------------------------------------------------------------')

    return average_acc

def main():
    mode = 3 #0 = transfer_learning, 1 = single_patient, 2 = all_patients, 3 = transfer_learning vs non_transfer learning (single_patient)

    K = 2 # number of folds for crossvalidation 


    patient_objects, labelset = select_patients(utils.pns, 5) #all patients with at least 5 classes
    general_batch, labelset = gen_batch(patient_objects, labelset, 100, ds=0.8) #all classes with at least 100 samples
    print(labelset)
    print([p.number for p in patient_objects])
    print(len(general_batch))

    # Set desired architecture
    hp = [32, 7, 3, 0.5, 75, 0.1, 'categorical_crossentropy']
    
    if mode == 0: #transfer learning
        general_patient_data, general_patient_data_labels, single_patient_data, single_patient_data_labels = next(generate_training_batches(patient_objects, general_batch, labelset))
        output_size = len(labelset)
        general_model = generate_trained_general_model(general_patient_data, general_patient_data_labels, hp, output_size)
        for layer in general_model.layers[:4]:
            layer.trainable = False
        for layer in general_model.layers:
            print(layer, layer.trainable)
        k_fold_crossvalidation_training(single_patient_data, single_patient_data_labels, hp, output_size, K, general_model)
    if mode == 1: # single patient
        general_patient_data, general_patient_data_labels, single_patient_data, single_patient_data_labels = next(generate_training_batches(patient_objects, general_batch, labelset))
        output_size = len(labelset)
        k_fold_crossvalidation_training(single_patient_data, single_patient_data_labels, hp, output_size, K)
    if mode == 2: # all patients
        general_patient_data, general_patient_data_labels, single_patient_data, single_patient_data_labels = next(generate_training_batches(patient_objects, general_batch, labelset))
        output_size = len(labelset)
        k_fold_crossvalidation_training(general_patient_data, general_patient_data_labels, hp, K, output_size)
    if mode == 3: # comparing transfer learning with non-transfer learning
        acc_transfer_learning = []
        acc_non_transfer_learning = []
        for general_patient_data, general_patient_data_labels, single_patient_data, single_patient_data_labels in generate_training_batches(patient_objects, general_batch, labelset):
            output_size = len(labelset)
            # transfer learning
            general_model = generate_trained_general_model(general_patient_data, general_patient_data_labels, hp, output_size)
            number_of_frozen_layers = 5
            for layer in general_model.layers[:number_of_frozen_layers]:
                layer.trainable = False
            acc_transfer_learning.append(k_fold_crossvalidation_training(single_patient_data, single_patient_data_labels, hp, output_size, K, general_model, verbose=0))
            # non-transfer learning
            acc_non_transfer_learning.append(k_fold_crossvalidation_training(single_patient_data, single_patient_data_labels, hp, output_size, K, verbose=0))
        avrg_acc_tl = np.mean(acc_transfer_learning)
        avrg_acc_tl_std = np.std(acc_transfer_learning)
        avrg_acc_ntl = np.mean(acc_non_transfer_learning)
        avrg_acc_ntl_std = np.std(acc_non_transfer_learning)
        print('------------------------------------------------------------------------')
        print(f'Number of cross-validation folds: {K}')
        print(f'Number of frozen layers: {number_of_frozen_layers}')
        print(f'Average transfer learning accuracy: {avrg_acc_tl} +- {avrg_acc_tl_std}')
        print(f'Average non_transfer learning accuracy: {avrg_acc_ntl} +- {avrg_acc_ntl_std}')
        print('------------------------------------------------------------------------')
        plot_box_and_whisker(avrg_acc_tl, avrg_acc_ntl)


        
    

if __name__ == "__main__":
    main()
