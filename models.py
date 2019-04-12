import os
import pickle
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


def pickle_it(variable, filename):

    with open(filename, "wb") as f:
        pickle.dump(variable, f)

    print("Pickled to ... " + filename)


def save_model(model, model_name):
    model_json = model.to_json()

    with open(os.path.join("Models", model_name + ".json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join("Models", model_name + ".h5"))
    print("Saved model to disk")


def load_model(model_name):
    # load json and create model
    json_file = open(os.path.join("Models", model_name + ".json"), "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")


if __name__=="__main__":
    pass
