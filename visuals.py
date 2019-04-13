import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_epoch_accuracy(train, test=None, filename="plot.png"):
    df = pd.DataFrame({'x': range(1, len(train)+1), 'train': train, 'val': test})
    plt.plot('x', 'train', data=df, linewidth=2, color='blue', label='train')
    plt.plot('x', 'val', data=df, linewidth=2, color='red', label='test')
    plt.savefig(os.path.join("Graphs", filename))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc="upper left")
    print("Graph created")


def write_history_csv(history, filename="history.csv"):

    with open(os.path.join("Graphs", filename), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=range(1, len(history.history['acc'])+1))
        writer.writeheader()
        for data in history:
            writer.writerow(data)

    print('CSV file written')

#def calc_confusion_matrix(model, data):

    #pred=model.predict_generator(data.generate_testing_data(32),100)
   #data.

   # matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))


if __name__ == "__main__":
    pass
