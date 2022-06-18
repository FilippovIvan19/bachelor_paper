import traceback
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sn
# sn.set(font_scale=3.0)

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


def draw_confusion_matrix(y_true, y_pred, history_dir, model_name):
    matrix = confusion_matrix(y_true, y_pred)
    matrix_norm = confusion_matrix(y_true, y_pred, normalize='true')

    file_name = history_dir + 'matrix'
    file_name_norm = history_dir + 'matrix_norm'

    plt.figure()
    ax = sn.heatmap(matrix, annot=True, cmap='Blues')
    ax.set_title(model_name)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

    plt.figure()
    ax = sn.heatmap(matrix_norm, annot=True, cmap='Blues')
    ax.set_title(model_name)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    plt.savefig(file_name_norm, bbox_inches='tight')
    plt.close()


def draw_history_graph(hist, history_dir):
    plot_epochs_metric(hist, history_dir + 'loss', 'loss')
    plot_epochs_metric(hist, history_dir + 'recall', 'recall')


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist[metric])
    plt.plot(hist['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def reformat_data(train_test_data, short=False):
    x_train, y_train, x_test, y_test = train_test_data
    input_shape = x_train[0].shape
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    encoder = OneHotEncoder(categories='auto')
    encoder.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_test_labeled = y_test.copy()
    y_train = encoder.transform(y_train.reshape(-1, 1)).toarray()
    y_test = encoder.transform(y_test.reshape(-1, 1)).toarray()

    if short:
        return x_train[:16], y_train[:16], x_test[:4], y_test[:4], y_test_labeled[:4], encoder, input_shape, nb_classes
    else:
        return x_train, y_train, x_test, y_test, y_test_labeled, encoder, input_shape, nb_classes


def print_metrics(dataset_name, model_name, metrics):
    print('    dataset {} was processed by {} model with following metrics:'.format(dataset_name, model_name))
    print(PRINT_METRICS_STRING.format(*metrics))


def print_exception(dataset_name, model_name, logf):
    print('DATASET={} MODEL={}'.format(dataset_name, model_name), file=logf)
    print(traceback.format_exc(), file=logf)
    print(file=logf)
    print('    dataset {} was NOT processed by {} model. Exception was caught\n'
          .format(dataset_name, model_name))


def save_metrics_to_xlsx(xlsx_file_name, stored_metrics_dfs):
    writer = pd.ExcelWriter(xlsx_file_name, engine='xlsxwriter')
    for ds_name, df in stored_metrics_dfs.items():
        df.to_excel(writer, sheet_name=ds_name, index=False, header=COLUMN_NAMES)
    writer.save()
