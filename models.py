from keras.layers import Input, Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.layers.advanced_activations import ReLU
from keras.models import Sequential
from keras.optimizers import SGD, rmsprop, Adam

nb_classes = 40


def model_cnn5313_dense3():
    model = Sequential()
    model.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(200, 200, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(8192, activation='relu'))  # make it the size of what comes out of the flatten
    model.add(Dropout(0.2))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes, activation='softmax'))

    return model


def model_cnn53133_dense3_batchnorm():

    # removing batch norm on dense gives higher performance on hyper params over 30 epoch. Maybe better over 50.
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(200, 200, 1)))
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
    model.add(Flatten())
    model.add(Dense(4608, activation='relu'))  # make it the size of what comes out of the flatten
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes, activation='softmax'))

    return model
