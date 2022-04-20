import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import OneHotEncoder

from src.constants import COLUMN_NAMES, PRINT_METRICS_STRING

matplotlib.use('agg')


def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return precision, accuracy, recall


def draw_history_graph(hist, history_dir):
    plot_epochs_metric(hist, history_dir + 'loss', 'loss')
    plot_epochs_metric(hist, history_dir + 'recall', 'recall')


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def reformat_data(train_test_data):
    x_train, y_train, x_test, y_test = train_test_data
    input_shape = x_train[0].shape
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    encoder = OneHotEncoder(categories='auto')
    encoder.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_test_labeled = y_test.copy()
    y_train = encoder.transform(y_train.reshape(-1, 1)).toarray()
    y_test = encoder.transform(y_test.reshape(-1, 1)).toarray()

    return x_train, y_train, x_test, y_test, y_test_labeled, encoder, input_shape, nb_classes


def print_metrics(dataset_name, model_name, metrics):
    print('    dataset {} was processed by {} model with following metrics:'.format(dataset_name, model_name))
    print(PRINT_METRICS_STRING.format(*metrics))


def save_metrics_to_xlsx(xlsx_file_name, stored_metrics_dfs):
    writer = pd.ExcelWriter(xlsx_file_name)
    for ds_name, df in stored_metrics_dfs.items():
        df.to_excel(writer, sheet_name=ds_name, index=False, header=COLUMN_NAMES)
    writer.save()
