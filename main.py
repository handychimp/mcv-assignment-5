import os
from datagenerators import DataGenerator

import pandas as pd
import numpy as np
import matplotlib
import models as m
import visuals as v
from keras.layers import Input, Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, BatchNormalization
from keras.layers.advanced_activations import ReLU
from keras.models import Sequential
from keras.optimizers import SGD, rmsprop, Adam
from sklearn.model_selection import train_test_split
from keras import backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# session = tf.Session(config=config)
# session.run()

np.random.seed(1337)  # for reproducibility

batch_size = 32
nb_classes = 40
nb_epoch = 30


if __name__ == "__main__":

    data = DataGenerator(image_size=200)

    with tf.Session(config=config) as sess:
        model = Sequential()
        model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(200, 200, 1)))
        # model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(200, 200, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Conv2D(256, kernel_size=(1, 1), strides=(1, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
        # model.add(Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4608, activation='relu')) # make it the size of what comes out of the flatten
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(2048, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(2048, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(nb_classes, activation='softmax'))

        # gradient decent parameters
        # sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
        # rms = rmsprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
        # ad = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        # model.compile(optimizer=sgd,
        #              loss='categorical_crossentropy',
        #              metrics=['accuracy'])

        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


        # print a summary of the model
        model.summary()
        l_rates = [0.01, 0.02, 0.05, 0.075, 0.1, 0.16, 0.2, 0.3]
        accs = []
        for rate in l_rates:
            sgd = SGD(lr=rate, decay=1e-6, momentum=0.9, nesterov=True)
            # fit the model with our created generator, validate on the generator with the validation data over 10 batches
            history = model.fit_generator(data.generate_training_data(batch_size), epochs=nb_epoch,
                                          steps_per_epoch=64, verbose=1,
                                          validation_data=data.generate_testing_data(batch_size),
                                          validation_steps=10)

            # score by doing a full run over the validation set
            score = model.evaluate_generator(data.generate_testing_data(batch_size), verbose=0, steps=15)
            model_name = "model_lr" + str(rate).replace('.', '')
            v.plot_epoch_accuracy(history.history['acc'], history.history['val_acc'], model_name)
            accs.append(score[1])

        idx = accs.index(max(accs))
        lr = l_rates[idx]

        momentums = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]

        accs=[]
        for mom in momentums:
            sgd = SGD(lr=lr, decay=1e-6, momentum=mom, nesterov=True)
            # fit the model with our created generator, validate on the generator with the validation data over 10 batches
            history = model.fit_generator(data.generate_training_data(batch_size), epochs=nb_epoch,
                                          steps_per_epoch=64, verbose=1,
                                          validation_data=data.generate_testing_data(batch_size),
                                          validation_steps=10)

            # score by doing a full run over the validation set
            score = model.evaluate_generator(data.generate_testing_data(batch_size), verbose=0, steps=15)
            model_name = "model_mom" + str(mom).replace('.', '')
            v.plot_epoch_accuracy(history.history['acc'], history.history['val_acc'], model_name)
            accs.append(score[1])

        idx = accs.index(max(accs))
        momentum = momentums[idx]

    # m.save_model(model, "model0")
    # m.pickle_it(history, "model0_output")

    # v.plot_epoch_accuracy(history.history['acc'], history.history['val_acc'], "plot_model0.png")
    # v.write_history_csv(history,"history.csv")

    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
    # print(data.classes)
    # decays = [1e-6]

    print("momentum: " + str(momentum))
    print("learning rate: " + str(rate))
