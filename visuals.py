import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras import utils
import datetime
import shutil
import glob

import copy

GRAPHS_FOLDER = "RunsData"
CODE_FOLDER = "Code"
MODELS_FOLDER = "Models"

date_time = datetime.datetime.now()
run_id = "{:04d}-{:02d}-{:02d}_{:02d}_{:02d}@{}".format(
    date_time.year,
    date_time.month,
    date_time.day,
    date_time.hour,
    date_time.minute,
    os.environ["COMPUTERNAME"])

if not os.path.exists(GRAPHS_FOLDER):
    os.mkdir(GRAPHS_FOLDER)

if os.path.exists(os.path.join(GRAPHS_FOLDER, run_id)):
    run_id += "_new"

os.mkdir(os.path.join(GRAPHS_FOLDER, run_id))
os.mkdir(os.path.join(GRAPHS_FOLDER, run_id, CODE_FOLDER))
os.mkdir(os.path.join(GRAPHS_FOLDER, run_id, MODELS_FOLDER))




for file in glob.glob(r'*.py', recursive=False):
    shutil.copy2(file, os.path.join(GRAPHS_FOLDER, run_id, CODE_FOLDER))

print("Created folder '{}' for this run.".format(run_id))


def plot_epoch_accuracy(train, test=None, filename="plot.png"):
    plt.figure()

    df = pd.DataFrame({'x': range(1, len(train)+1), 'train': train, 'val': test})

    plt.plot('x', 'train', data=df, linewidth=2, color='blue', label='train')
    plt.plot('x', 'val', data=df, linewidth=2, color='red', label='test')
    plt.ylabel('accuracy')

    plt.xlabel('epoch')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(GRAPHS_FOLDER, run_id, filename))
    print("Graph '{}' created.".format(filename))
    del df


def plot_progressive(vals, filename='prog-epochs.png'):
    plt.figure()

    df = pd.DataFrame({'x': range(1, len(vals)+1), 'val': vals})

    plt.plot('x', 'val', data=df, linewidth=2, color='red', label='value')
    plt.ylabel('accuracy')

    plt.xlabel('layer')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(GRAPHS_FOLDER, run_id, filename))
    print("Graph '{}' created.".format(filename))
    del df

def csv_splitter(data_dict):
    data_dict = copy.deepcopy(data_dict)

    output = []
    while True:
        row = dict()

        for key, value in data_dict.items():
            if len(value) == 0:
                return output

            row[key] = value.pop(0)

        output.append(row)


def write_history_csv(history, filename="history.csv"):

    with open(os.path.join(GRAPHS_FOLDER, run_id,  filename), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=history.history.keys())
        writer.writeheader()

        data = csv_splitter(history.history)

        for row in data:
            writer.writerow(row)

    print("CSV file '{}' created.".format(filename))

def calc_confusion_matrix(model, data):

    pred=model.predict_generator(data.generate_testing_data(32), 100)

    cm = confusion_matrix(data.class_names, pred)





def plot_model(model, file_name='model.png'):
    utils.plot_model(model, to_file=os.path.join(GRAPHS_FOLDER, run_id, file_name))
    print("Model '{}' plotted.".format(file_name))


def write_summary(model, file_name='summary.txt'):
    with open(os.path.join(GRAPHS_FOLDER, run_id, file_name), "w+") as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    print("Summary '{}' written.".format(file_name))


def save_model(model, file_name='model.h5'):
    model.save(os.path.join(GRAPHS_FOLDER, run_id, MODELS_FOLDER, file_name))
    print("Model saved as '{}'.".format(file_name))


def model_path(file_name=None):
    if file_name is None:
        return os.path.join(GRAPHS_FOLDER, run_id, MODELS_FOLDER)
    else:
        return os.path.join(GRAPHS_FOLDER, run_id, MODELS_FOLDER, file_name)


if __name__ == "__main__":
    pass
