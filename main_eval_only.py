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

from src.utils import reformat_data, print_metrics, save_metrics_to_xlsx, print_exception
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
xlsx_file_name = result_dir + 'results_eval.xlsx'
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

        for model in MODELS_TO_RUN:
            try:
                classifier_class: Type[BaseClassifier] = get_classifier_class(model)
                history_dir = current_dir + HISTORY_DIR_SUFFIX + dataset_name + '/' + model.value + '/'
                classifier = classifier_class(data, model, history_dir, print_epoch_num=PRINT_EPOCH_NUM)

                dataset_metrics[model.value] = classifier.run_eval_only(model.value)
                stored_metrics_dfs[dataset_name] = pd.DataFrame.from_dict(dataset_metrics, orient='index').reset_index()
                save_metrics_to_xlsx(xlsx_file_name, stored_metrics_dfs)

                if PRINT_METRICS:
                    print_metrics(dataset_name, model.value, dataset_metrics[model.value])
            except Exception as e:
                if PRINT_METRICS:
                    print_exception(dataset_name, model.value, logf)

        stored_metrics_dfs[dataset_name] = pd.DataFrame.from_dict(dataset_metrics, orient='index').reset_index()

save_metrics_to_xlsx(xlsx_file_name, stored_metrics_dfs)
logf.close()

full_duration = time.time() - full_start_time
print('full duration = {:.2f} minutes'.format(full_duration/60))
