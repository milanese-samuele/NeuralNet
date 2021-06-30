import tensorflow as tf
from tensorflow.keras import datasets, layers, regularizers, models
import matplotlib.pyplot as plt

#prefetch?
#https://keras.io/examples/vision/image_classification_from_scratch/
#train_ds = train_ds.prefetch(buffer_size=32)
#val_ds = val_ds.prefetch(buffer_size=32)

def build_model(input_shape, hp):
    inputs = tf.keras.Input(shape=input_shape)
    # Rescale images to [0, 1] (z-score scalling)
    # x = Rescaling(scale=1.0 / 255)(x)
    
    # Apply some convolution and pooling layers
    x = layers.Conv1D(filters=16, kernel_size=13, activation="relu")(inputs)
    x = layers.AveragePooling1D(pool_size=3)(x)
    x = layers.Conv1D(filters=15, kernel_size=32, activation="relu")(x)
    x = layers.AveragePooling1D(pool_size=3)(x)
    x = layers.Conv1D(filters=17, kernel_size=64, activation="relu")(x)
    x = layers.AveragePooling1D(pool_size=3)(x)
    x = layers.Conv1D(filters=19, kernel_size=128, activation="relu")(x)
    x = layers.AveragePooling1D(pool_size=3)(x)
    # tries [0.3, 0.5, 0.7] as dropout-rates
    x = layers.Dropout(0.5]))(x)
    x = layers.Dense(units=35, kernel_regularizer=regularizers.l2(5))(x)
    #l2 reg twice??
    x = layers.Dense(units=5, kernel_regularizer=regularizers.l2(5))(x)
    outputs = layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

def train_model(model):
    # specify an optimizer and a loss function 
    # differentiable loss function for the network training, try: quadratic loss, L1-norm loss, or logistic regression (probably a good candidate)
    model.compile(
                    optimizer='sgd', #stochastic gradient descent (try different ones)
                    loss='categorical_crossentropy', #try different ones
                    metrics=['acc']
    )
    
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]   

    # fitting the model to the data
    history = model.fit(
                        training_set_with_labels, 
                        validation_data=validation_set_with_labels,
                        batch_size=36, 
                        epochs=60,
                        callbacks=callbacks) #bs and epochs needs tuning
    
    # history contains per-epoch metrics values 
    print(history.history)

    #potentially add model saving

def evaluate_model(model, validation_dataset):
    los, acc = model.evaluate(validation_dataset)



