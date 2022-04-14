import abc
from typing import Type

import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy, from_multi_index_to_3d_numpy

from src.constants import Archives
import abc


class BaseAdapter(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def read_x_y_data(file_path: str) -> (np.array, np.array):
        return np.array(0), np.array(0)

    @classmethod
    @abc.abstractmethod
    def read_train_test_data(cls, archives_dir: str, dataset_name: str) -> (np.array, np.array, np.array, np.array):
        return np.array(0), np.array(0), np.array(0), np.array(0)


class UCR2018Adapter(BaseAdapter):
    @staticmethod
    def read_x_y_data(file_path: str) -> (np.array, np.array):
        data, labels = load_from_tsfile_to_dataframe(file_path)
        return data, labels

    @classmethod
    def read_train_test_data(cls, archives_dir: str, dataset_name: str) -> (np.array, np.array, np.array, np.array):
        file_pre_path = archives_dir + Archives.UCR_2018.value + '/' + dataset_name + '/' + dataset_name
        x_train, y_train = cls.read_x_y_data(file_pre_path + '_TRAIN.ts')
        x_test, y_test = cls.read_x_y_data(file_pre_path + '_TEST.ts')
        x_train = from_nested_to_3d_numpy(x_train).swapaxes(1, 2)
        x_test = from_nested_to_3d_numpy(x_test).swapaxes(1, 2)
        return x_train, y_train, x_test, y_test


archive_names_to_adapters = {
    Archives.UCR_2018: UCR2018Adapter
}


def get_adapter(archive: Archives) -> Type[BaseAdapter]:
    return archive_names_to_adapters[archive]
