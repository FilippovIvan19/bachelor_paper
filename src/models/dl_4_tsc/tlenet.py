# t-leNet model: t-leNet + WW 
import tensorflow as tf
from tensorflow import keras
import numpy as np

from src.models.base_model import BaseModel


class Model_TLENET(BaseModel):
    def __init__(self, input_shape, nb_classes):
        self.callbacks = []
        self.batch_size = 256
        # self.nb_epochs = 1000
        self.nb_epochs = 100

        self.warping_ratios = [0.5, 1, 2]
        self.slice_ratio = 0.1
        self.tot_increase_num = 0

        self.nb_classes = nb_classes
        self.model = None

    def slice_data(self, data_x, data_y, length_sliced):
        # print('data_x.shape =', data_x.shape)
        # print('length_sliced =', length_sliced)
        n = data_x.shape[0]
        length = data_x.shape[1]
        n_dim = data_x.shape[2]  # for MTS
        nb_classes = data_y.shape[1]

        increase_num = length - length_sliced + 1  # if increase_num =5, it means one ori becomes 5 new instances.
        n_sliced = n * increase_num

        # print((n_sliced, length_sliced, n_dim))

        new_x = np.zeros((n_sliced, length_sliced, n_dim))
        new_y = np.zeros((n_sliced, nb_classes))
        for i in range(n):
            for j in range(increase_num):
                new_x[i * increase_num + j, :, :] = data_x[i, j: j + length_sliced, :]
                new_y[i * increase_num + j] = np.int_(data_y[i].astype(np.float32))

        return new_x, new_y, increase_num

    def window_warping(self, data_x, warping_ratio):
        num_x = data_x.shape[0]
        len_x = data_x.shape[1]
        dim_x = data_x.shape[2]

        x = np.arange(0, len_x, warping_ratio)
        xp = np.arange(0, len_x)

        new_length = len(np.interp(x, xp, data_x[0, :, 0]))

        warped_series = np.zeros((num_x, new_length, dim_x), dtype=np.float64)

        for i in range(num_x):
            for j in range(dim_x):
                warped_series[i, :, j] = np.interp(x, xp, data_x[i, :, j])

        return warped_series

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        conv_1 = keras.layers.Conv1D(filters=5, kernel_size=5, activation='relu', padding='same')(input_layer)
        conv_1 = keras.layers.MaxPool1D(pool_size=2)(conv_1)

        conv_2 = keras.layers.Conv1D(filters=20, kernel_size=5, activation='relu', padding='same')(conv_1)
        conv_2 = keras.layers.MaxPool1D(pool_size=4)(conv_2)

        # they did not mention the number of hidden units in the fully-connected layer
        # so we took the lenet they referenced 

        flatten_layer = keras.layers.Flatten()(conv_2)
        fully_connected_layer = keras.layers.Dense(500, activation='relu')(flatten_layer)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(fully_connected_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01, decay=0.005),
                      loss='categorical_crossentropy', metrics=[keras.metrics.Recall()])

        return model

    def pre_processing(self, x_train, y_train, x_test, y_test):
        length_ratio = int(self.slice_ratio * x_train.shape[1])

        x_train_augmented = []  # list of the augmented as well as the original data
        x_test_augmented = []  # list of the augmented as well as the original data

        y_train_augmented = []
        y_test_augmented = []

        # data augmentation using WW
        for warping_ratio in self.warping_ratios:
            x_train_augmented.append(self.window_warping(x_train, warping_ratio))
            x_test_augmented.append(self.window_warping(x_test, warping_ratio))
            y_train_augmented.append(y_train)
            y_test_augmented.append(y_test)

        increase_nums = []

        # data augmentation using WS 
        for i in range(0, len(x_train_augmented)):
            x_train_augmented[i], y_train_augmented[i], increase_num = self.slice_data(
                x_train_augmented[i], y_train, length_ratio)
            x_test_augmented[i], y_test_augmented[i], increase_num = self.slice_data(
                x_test_augmented[i], y_test, length_ratio)
            increase_nums.append(increase_num)

        tot_increase_num = np.array(increase_nums).sum()
        self.tot_increase_num = tot_increase_num

        new_x_train = np.zeros((x_train.shape[0] * tot_increase_num, length_ratio, x_train.shape[2]))
        new_y_train = np.zeros((y_train.shape[0] * tot_increase_num, y_train.shape[1]))

        new_x_test = np.zeros((x_test.shape[0] * tot_increase_num, length_ratio, x_test.shape[2]))
        new_y_test = np.zeros((y_test.shape[0] * tot_increase_num, y_test.shape[1]))

        # merge the list of augmented data
        idx = 0
        for i in range(x_train.shape[0]):
            for j in range(len(increase_nums)):
                increase_num = increase_nums[j]
                new_x_train[idx:idx + increase_num, :, :] = \
                    x_train_augmented[j][i * increase_num:(i + 1) * increase_num, :, :]
                new_y_train[idx:idx + increase_num, :] = \
                    y_train_augmented[j][i * increase_num:(i + 1) * increase_num, :]
                idx += increase_num

        # do the same for the test set 
        idx = 0
        for i in range(x_test.shape[0]):
            for j in range(len(increase_nums)):
                increase_num = increase_nums[j]
                new_x_test[idx:idx + increase_num, :, :] = \
                    x_test_augmented[j][i * increase_num:(i + 1) * increase_num, :, :]
                new_y_test[idx:idx + increase_num, :] = \
                    y_test_augmented[j][i * increase_num:(i + 1) * increase_num, :]
                idx += increase_num
        return new_x_train, new_y_train, new_x_test, new_y_test

    def tlenet_predict(self, x_test, model_path):
        model = keras.models.load_model(model_path)

        y_pred = model.predict(x_test, batch_size=self.batch_size)
        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        # get the true predictions of the test set
        y_predicted = []
        test_num_batch = int(x_test.shape[0] / self.tot_increase_num)
        for i in range(test_num_batch):
            unique_value, sub_ind, correspond_ind, count = np.unique(y_pred, True, True, True)

            idx_max = np.argmax(count)
            predicted_label = unique_value[idx_max]

            y_predicted.append(predicted_label)

        y_pred = np.array(y_predicted)
        return y_pred

    def prepare(self, x_train, y_train, x_test, y_test):
        x_train = x_train.swapaxes(1, 2)
        x_test = x_test.swapaxes(1, 2)
        # limit the number of augmented time series if series too long or too many
        if x_train.shape[1] > 500 or x_train.shape[0] > 2000 or x_test.shape[0] > 2000:
            self.warping_ratios = [1]
            self.slice_ratio = 0.9
        # increase the slice if series too short
        if x_train.shape[1] * self.slice_ratio < 8:
            self.slice_ratio = 8 / x_train.shape[1]

        x_train, y_train, x_test, y_test = self.pre_processing(x_train, y_train, x_test, y_test)
        input_shape = x_train.shape[1:]

        self.model = self.build_model(input_shape, self.nb_classes)
        return x_train, y_train, x_test, y_test
