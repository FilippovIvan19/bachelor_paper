import time; full_start_time = time.time()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*pandas.Int64Index is deprecated.*")
warnings.filterwarnings("ignore", ".*Set a lower value for log_every_n_steps if you want to see logs.*")
warnings.filterwarnings("ignore", ".*order of the arguments: ceil_mode and return_indices.*")

import pytorch_lightning
import logging; logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)

import pandas as pd
from typing import Type

from src.utils import reformat_data, print_metrics, save_metrics_to_xlsx, print_exception, calculate_metrics
from src.utils import draw_confusion_matrix
from src.classifiers import get_classifier_class, BaseClassifier
from src.constants import ARCHIVES_DIR_SUFFIX, HISTORY_DIR_SUFFIX, RESULTS_DIR_SUFFIX, COLUMN_NAMES
from src.constants import check_archive_contains_dataset
from src.adapters import get_adapter, BaseAdapter
from run_config import *


check_archive_contains_dataset(DATASETS_TO_RUN)

current_dir = os.path.abspath(os.getcwd())
result_dir = current_dir + RESULTS_DIR_SUFFIX
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
xlsx_file_name = result_dir + 'results.xlsx'
logf = open(result_dir + 'exceptions.log', 'w')

stored_metrics_dfs = dict()
if os.path.exists(xlsx_file_name) and os.path.getsize(xlsx_file_name) != 0:
    stored_metrics_dfs = pd.read_excel(xlsx_file_name, sheet_name=None)

for archives in DATASETS_TO_RUN.items():
    cur_archive = archives[0]
    cur_datasets = archives[1]
    for dataset_name in cur_datasets:
        dataset_metrics = dict()
        if dataset_name in stored_metrics_dfs:
            dataset_metrics = stored_metrics_dfs[dataset_name].set_index(COLUMN_NAMES[0]).T.to_dict('list')

        adapter: Type[BaseAdapter] = get_adapter(cur_archive)
        train_test_data = adapter.read_train_test_data(
            current_dir + ARCHIVES_DIR_SUFFIX + cur_archive.value, dataset_name)
        if PRINT_READING:
            print('dataset {} from archive {} was read'.format(dataset_name, cur_archive.value))
        data = reformat_data(train_test_data, SHORT_DATA)

        sum_probs = None
        classifier = None
        for model in MODELS_FOR_ENSEMBLE:
            try:
                classifier_class: Type[BaseClassifier] = get_classifier_class(model)
                history_dir = current_dir + HISTORY_DIR_SUFFIX + dataset_name + '/' + model.value + '/'
                classifier = classifier_class(data, model, history_dir, print_epoch_num=PRINT_EPOCH_NUM)

                probs = classifier.predict_probabilities()
                if sum_probs is None:
                    sum_probs = probs
                else:
                    sum_probs += probs
            except Exception as e:
                if PRINT_METRICS:
                    print_exception(dataset_name, model.value, logf)

        y_predicted_labeled = classifier.encoder.inverse_transform(sum_probs)

        precision, accuracy, recall = calculate_metrics(classifier.y_test_labeled, y_predicted_labeled)

        history_dir = current_dir + HISTORY_DIR_SUFFIX + dataset_name + '/' + ENS_NAME + '/'
        if ~os.path.exists(history_dir):
            os.makedirs(history_dir)
        draw_confusion_matrix(classifier.y_test_labeled, y_predicted_labeled, history_dir, ENS_NAME)

        dataset_metrics[ENS_NAME] = [precision, accuracy, recall, 0]

        stored_metrics_dfs[dataset_name] = pd.DataFrame.from_dict(dataset_metrics, orient='index').reset_index()
        save_metrics_to_xlsx(xlsx_file_name, stored_metrics_dfs)
        if PRINT_METRICS:
            print_metrics(dataset_name, ENS_NAME, dataset_metrics[ENS_NAME])

        stored_metrics_dfs[dataset_name] = pd.DataFrame.from_dict(dataset_metrics, orient='index').reset_index()

save_metrics_to_xlsx(xlsx_file_name, stored_metrics_dfs)
logf.close()

full_duration = time.time() - full_start_time
print('full duration = {:.2f} minutes'.format(full_duration/60))
