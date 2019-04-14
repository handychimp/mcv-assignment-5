import os
from datagenerators import DataGenerator
from keras.applications import xception
import pandas as pd
import numpy as np
import matplotlib
import models as m
import visuals as v
from keras.layers import Input, Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, BatchNormalization
from keras.layers.advanced_activations import ReLU
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD, rmsprop, Adam
from sklearn.model_selection import train_test_split
from keras import backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.callbacks import ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
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

image_size = 150


def progressive_layer_train(pl_model, pl_epoch, pl_batches):
    current_depth = 0

    global sgd
    global history

    pl_model.save_weights('temp.h5')

    values = []

    while current_depth < len(model.layers):

        pl_model.load_weights('temp.h5')

        cur_layer = 0

        for pl_layer in pl_model.layers:
            pl_layer.trainable = cur_layer < current_depth
            cur_layer += 1

        print("Training all layers")

        pl_checkpoint = ModelCheckpoint(v.model_path('progressive-layers-depth-' +
                                                     current_depth + '-{epoch:03d}-{val_acc:.2f}.hdf5'),
                                        monitor='val_acc',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='max')
        pl_callbacks_list = [pl_checkpoint]

        sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        history = model.fit_generator(data.generate_training_data(pl_batches), epochs=pl_epoch,
                                      steps_per_epoch=64, verbose=1,
                                      validation_data=data.generate_testing_data(pl_batches),
                                      validation_steps=10,
                                      callbacks=pl_callbacks_list)

        values.append(max(history.history['val_acc']))

        v.plot_epoch_accuracy(history.history['acc'], history.history['val_acc'], 'depth-{}-plot')

    v.plot_progressive()


if __name__ == "__main__":

    data = DataGenerator(image_size=image_size)

    with tf.Session(config=config) as sess:
        xc_model = xception.Xception(
            include_top=False,
            weights='imagenet',
            input_shape=(image_size, image_size, 3))

        x = xc_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(nb_classes, activation='softmax')(x)

        model = Model(xc_model.input, predictions)

        # gradient decent parameters
        # sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
        # rms = rmsprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
        # ad = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        # model.compile(optimizer=sgd,
        #              loss='categorical_crossentropy',
        #              metrics=['accuracy'])

        for layer in xc_model.layers:
            layer.trainable = False

        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

        v.plot_model(model)

        model.summary()
        v.write_summary(model)

        print("Training only top layers quickly")

        tl_epoch = 2
        tl_batch = 32

        sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        history = model.fit_generator(data.generate_training_data(tl_batch), epochs=tl_epoch,
                                      steps_per_epoch=64, verbose=1,
                                      validation_data=data.generate_testing_data(batch_size),
                                      validation_steps=10)

        top_model_name = "only-top-layers_{}-epochs_{}-batch-size".format(tl_epoch, tl_batch)
        v.plot_epoch_accuracy(history.history['acc'], history.history['val_acc'], top_model_name)

        for layer in xc_model.layers:
            layer.trainable = True

        v.write_history_csv(history, top_model_name + ".csv")

        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

        print("Training all layers")

        checkpoint = ModelCheckpoint(v.model_path('weights-improvement-{epoch:03d}-{val_acc:.2f}.hdf5'), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        history = model.fit_generator(data.generate_training_data(batch_size), epochs=nb_epoch,
                                      steps_per_epoch=64, verbose=1,
                                      validation_data=data.generate_testing_data(batch_size),
                                      validation_steps=10,
                                      callbacks=callbacks_list)

        model_name = "all-layers_{}-epochs_{}-batch-size".format(nb_epoch, batch_size)
        v.plot_epoch_accuracy(history.history['acc'], history.history['val_acc'], model_name)

        v.write_history_csv(history, model_name + ".csv")

        score = model.evaluate_generator(data.generate_testing_data(batch_size), verbose=0, steps=15)

    # m.save_model(model, "model0")
    # m.pickle_it(history, "model0_output")

    # v.plot_epoch_accuracy(history.history['acc'], history.history['val_acc'], "plot_model0.png")
    # v.write_history_csv(history,"history.csv")

    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
    # print(data.classes)
    # decays = [1e-6]

    #print("momentum: " + str(momentum))
    #print("learning rate: " + str(rate))
