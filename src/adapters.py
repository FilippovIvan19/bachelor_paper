import abc
from typing import Type
import numpy as np
from sklearn.model_selection import train_test_split
from sktime.datasets import load_from_tsfile_to_dataframe
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
import wfdb

from src.constants import Archives


class BaseAdapter(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def read_train_test_data(cls, archive_dir: str, dataset_name: str) -> (np.array, np.array, np.array, np.array):
        return np.array(0), np.array(0), np.array(0), np.array(0)


class TSCAdapter(BaseAdapter):
    @staticmethod
    def read_x_y_data(file_path: str) -> (np.array, np.array):
        data, labels = load_from_tsfile_to_dataframe(file_path)
        return data, labels

    @classmethod
    def read_train_test_data(cls, archive_dir: str, dataset_name: str) -> (np.array, np.array, np.array, np.array):
        file_pre_path = archive_dir + '/' + dataset_name + '/' + dataset_name
        x_train, y_train = cls.read_x_y_data(file_pre_path + '_TRAIN.ts')
        x_test, y_test = cls.read_x_y_data(file_pre_path + '_TEST.ts')
        x_train = from_nested_to_3d_numpy(x_train).swapaxes(1, 2)
        x_test = from_nested_to_3d_numpy(x_test).swapaxes(1, 2)
        return x_train, y_train, x_test, y_test


class PTBAdapter(BaseAdapter):
    @staticmethod
    def comments_to_dict(comments):
        key_value_pairs = [comment.split(':') for comment in comments]
        return {pair[0]: pair[1] for pair in key_value_pairs}

    @classmethod
    def read_train_test_data(cls, archive_dir: str, dataset_name: str) -> (np.array, np.array, np.array, np.array):
        file_pre_path = archive_dir + '/' + dataset_name + '/'
        record_names = wfdb.io.get_record_list('ptbdb')  # [:5]
        data = []
        labels = []

        for record_name in record_names:
            path = file_pre_path + record_name
            record = wfdb.io.rdrecord(record_name=path)

            label = cls.comments_to_dict(record.comments)['Reason for admission'][1:]
            labels.append(label)
            record_data = wfdb.rdsamp(record_name=path)[0]  # time_length x series_count
            data.append(record_data[:, :15])

        x_train, x_test, y_train, y_test = train_test_split(
            data, np.array(labels), test_size=0.2, shuffle=True
        )

        return x_train, y_train, x_test, y_test


archive_names_to_adapters = {
    Archives.UCR_2018: TSCAdapter,
    Archives.UEA_2018: TSCAdapter,
    Archives.PTB: PTBAdapter
}


def get_adapter(archive: Archives) -> Type[BaseAdapter]:
    return archive_names_to_adapters[archive]
