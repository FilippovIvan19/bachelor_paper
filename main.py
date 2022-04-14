import time; full_start_time = time.time()
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings; warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from typing import Type

from src.utils import reformat_data, print_metrics
from src.classifiers import get_classifier_class, BaseClassifier
from src.constants import ARCHIVES_DIR_SUFFIX, HISTORY_DIR_SUFFIX, RESULTS_DIR_SUFFIX, COLUMN_NAMES
from src.constants import check_archive_contains_dataset
from src.adapters import get_adapter, BaseAdapter
from run_config import *


check_archive_contains_dataset(DATASETS_TO_RUN)

current_dir = os.path.abspath(os.getcwd())
xlsx_file_name = current_dir + RESULTS_DIR_SUFFIX + 'results.xlsx'

stored_metrics_dfs = dict()
if os.path.exists(xlsx_file_name):
    stored_metrics_dfs = pd.read_excel(xlsx_file_name, sheet_name=None)


for archives in DATASETS_TO_RUN.items():
    cur_archive = archives[0]
    cur_datasets = archives[1]
    for dataset_name in cur_datasets:
        csv_file_name = current_dir + RESULTS_DIR_SUFFIX + dataset_name + '.csv'
        dataset_metrics = dict()
        if os.path.exists(csv_file_name):
            dataset_metrics = pd.read_csv(csv_file_name).set_index(COLUMN_NAMES[0]).T.to_dict('list')

        adapter: Type[BaseAdapter] = get_adapter(cur_archive)
        train_test_data = adapter.read_train_test_data(current_dir + ARCHIVES_DIR_SUFFIX, dataset_name)
        if PRINT_READING:
            print('dataset {} from archive {} was read'.format(dataset_name, cur_archive.value))
        data = reformat_data(train_test_data)

        for model in MODELS_TO_RUN:
            classifier_class: Type[BaseClassifier] = get_classifier_class(model)
            history_dir = current_dir + HISTORY_DIR_SUFFIX + dataset_name + '/' + model.value + '/'
            classifier = classifier_class(data, model, history_dir)

            dataset_metrics[model.value] = classifier.run(
                save_history=SAVE_HISTORY, save_model=SAVE_MODEL, draw_graph=DRAW_GRAPH
            )

            if PRINT_METRICS:
                print_metrics(dataset_name, model.value, dataset_metrics[model.value])

        dataset_metrics_df = pd.DataFrame.from_dict(dataset_metrics, orient='index').reset_index()
        dataset_metrics_df.to_csv(csv_file_name, index=False, header=COLUMN_NAMES)
        stored_metrics_dfs[dataset_name] = dataset_metrics_df

writer = pd.ExcelWriter(xlsx_file_name)

for ds_name, df in stored_metrics_dfs.items():
    df.to_excel(writer, sheet_name=ds_name, index=False, header=COLUMN_NAMES)

writer.save()

full_duration = time.time() - full_start_time
print('full duration = {:.2f} minutes'.format(full_duration/60))
