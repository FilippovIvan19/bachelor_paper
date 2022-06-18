import tensorflow_addons as tfa
from tensorflow import keras
import torch.nn as nn
import numpy as np
import torch

from src.models.base_model import BaseModel
from src.models.lit_model import LitModule
from src.models.lxdv.author_models1d import Flatten


class Model_ENCODER_NO_POOLING(BaseModel):
    def __init__(self, input_shape, nb_classes):
        self.callbacks = []
        self.batch_size = 12
        # self.nb_epochs = 100
        self.nb_epochs = 250

        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape[::-1])

        # conv block -1
        conv1 = keras.layers.Conv1D(filters=128, kernel_size=5, strides=2)(input_layer)
        conv1 = tfa.layers.InstanceNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=0.2)(conv1)
        # conv block -2
        conv2 = keras.layers.Conv1D(filters=256, kernel_size=11, strides=5)(conv1)
        conv2 = tfa.layers.InstanceNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=0.2)(conv2)
        # conv block -3
        conv3 = keras.layers.Conv1D(filters=512, kernel_size=21, strides=1, padding='same')(conv2)
        conv3 = tfa.layers.InstanceNormalization()(conv3)
        conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = keras.layers.Dropout(rate=0.2)(conv3)
        # split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:, :, :256])(conv3)
        attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 256:])(conv3)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])
        # last layer
        dense_layer = keras.layers.Dense(units=256, activation='sigmoid')(multiply_layer)
        dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
        # output layer
        flatten_layer = keras.layers.Flatten()(dense_layer)
        output_layer = keras.layers.Dense(units=nb_classes, activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.00001),
                      metrics=[keras.metrics.Recall()])

        return model

    def prepare(self, x_train, y_train, x_test, y_test):
        return x_train.swapaxes(1, 2), y_train, x_test.swapaxes(1, 2), y_test


class Model_ENCODER_CONV_INSTEAD_POOLING(BaseModel):
    def __init__(self, input_shape, nb_classes):
        self.callbacks = []
        self.batch_size = 12
        # self.nb_epochs = 100
        self.nb_epochs = 250

        super().__init__(input_shape, nb_classes)

    def pool_with_conv(self, inp, filters):
        inp = keras.layers.Conv1D(filters=filters, kernel_size=2, strides=2)(inp)
        inp = tfa.layers.InstanceNormalization()(inp)
        inp = keras.layers.PReLU(shared_axes=[1])(inp)
        inp = keras.layers.Dropout(rate=0.2)(inp)
        return inp

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape[::-1])

        # conv block -1
        conv1 = keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same')(input_layer)
        conv1 = tfa.layers.InstanceNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=0.2)(conv1)
        conv1 = self.pool_with_conv(conv1, 128)
        # conv block -2
        conv2 = keras.layers.Conv1D(filters=256, kernel_size=11, strides=1, padding='same')(conv1)
        conv2 = tfa.layers.InstanceNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=0.2)(conv2)
        conv2 = self.pool_with_conv(conv2, 256)
        # conv block -3
        conv3 = keras.layers.Conv1D(filters=512, kernel_size=21, strides=1, padding='same')(conv2)
        conv3 = tfa.layers.InstanceNormalization()(conv3)
        conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = keras.layers.Dropout(rate=0.2)(conv3)
        # split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:, :, :256])(conv3)
        attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 256:])(conv3)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])
        # last layer
        dense_layer = keras.layers.Dense(units=256, activation='sigmoid')(multiply_layer)
        dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
        # output layer
        flatten_layer = keras.layers.Flatten()(dense_layer)
        output_layer = keras.layers.Dense(units=nb_classes, activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.00001),
                      metrics=[keras.metrics.Recall()])

        return model

    def prepare(self, x_train, y_train, x_test, y_test):
        return x_train.swapaxes(1, 2), y_train, x_test.swapaxes(1, 2), y_test


class Model_ENCODER_BATCH_NORM(BaseModel):
    def __init__(self, input_shape, nb_classes):
        self.callbacks = []
        self.batch_size = 12
        # self.nb_epochs = 100
        self.nb_epochs = 250

        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape[::-1])

        # conv block -1
        conv1 = keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=0.2)(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        # conv block -2
        conv2 = keras.layers.Conv1D(filters=256, kernel_size=11, strides=1, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=0.2)(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        # conv block -3
        conv3 = keras.layers.Conv1D(filters=512, kernel_size=21, strides=1, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = keras.layers.Dropout(rate=0.2)(conv3)
        # split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:, :, :256])(conv3)
        attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 256:])(conv3)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])
        # last layer
        dense_layer = keras.layers.Dense(units=256, activation='sigmoid')(multiply_layer)
        dense_layer = keras.layers.BatchNormalization()(dense_layer)
        # output layer
        flatten_layer = keras.layers.Flatten()(dense_layer)
        output_layer = keras.layers.Dense(units=nb_classes, activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.00001),
                      metrics=[keras.metrics.Recall()])

        return model

    def prepare(self, x_train, y_train, x_test, y_test):
        return x_train.swapaxes(1, 2), y_train, x_test.swapaxes(1, 2), y_test


class Model_ENCODER_EXTRA_LAYER(BaseModel):
    def __init__(self, input_shape, nb_classes):
        self.callbacks = []
        self.batch_size = 12
        # self.nb_epochs = 100
        self.nb_epochs = 250

        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape[::-1])

        # conv block -1
        conv1 = keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same')(input_layer)
        conv1 = tfa.layers.InstanceNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=0.2)(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        # conv block -2
        conv2 = keras.layers.Conv1D(filters=256, kernel_size=11, strides=1, padding='same')(conv1)
        conv2 = tfa.layers.InstanceNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=0.2)(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        # conv block -3
        conv3 = keras.layers.Conv1D(filters=512, kernel_size=16, strides=1, padding='same')(conv2)
        conv3 = tfa.layers.InstanceNormalization()(conv3)
        conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = keras.layers.Dropout(rate=0.2)(conv3)
        conv3 = keras.layers.MaxPooling1D(pool_size=2)(conv3)
        # conv block -4
        conv4 = keras.layers.Conv1D(filters=1024, kernel_size=21, strides=1, padding='same')(conv3)
        conv4 = tfa.layers.InstanceNormalization()(conv4)
        conv4 = keras.layers.PReLU(shared_axes=[1])(conv4)
        conv4 = keras.layers.Dropout(rate=0.2)(conv4)
        # split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:, :, :512])(conv4)
        attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 512:])(conv4)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])
        # last layer
        dense_layer = keras.layers.Dense(units=512, activation='sigmoid')(multiply_layer)
        dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
        # output layer
        flatten_layer = keras.layers.Flatten()(dense_layer)
        output_layer = keras.layers.Dense(units=nb_classes, activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.00001),
                      metrics=[keras.metrics.Recall()])

        return model

    def prepare(self, x_train, y_train, x_test, y_test):
        return x_train.swapaxes(1, 2), y_train, x_test.swapaxes(1, 2), y_test


class Model_INCEPTION_EXTRA_LAYER(BaseModel):
    def __init__(self, input_shape, nb_classes):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        self.callbacks = [reduce_lr]

        self.batch_size = 64
        # self.nb_epochs = 2
        self.nb_epochs = 800

        self.nb_filters = 32
        self.use_residual = True
        self.use_bottleneck = True
        self.depth = 9
        self.kernel_size = 41 - 1
        self.bottleneck_size = 32
        self.lr = 0.001

        super().__init__(input_shape, nb_classes)

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > self.bottleneck_size:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape[::-1])

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(self.lr),
                      metrics=[keras.metrics.Recall()])

        return model

    def prepare(self, x_train, y_train, x_test, y_test):
        self.batch_size = int(min(x_train.shape[0] / 10, self.batch_size))
        return x_train.swapaxes(1, 2), y_train, x_test.swapaxes(1, 2), y_test


class Model_INCEPTION_BIG_STEP(BaseModel):
    def __init__(self, input_shape, nb_classes):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        self.callbacks = [reduce_lr]

        self.batch_size = 64
        # self.nb_epochs = 2
        self.nb_epochs = 800

        self.nb_filters = 32
        self.use_residual = True
        self.use_bottleneck = True
        self.depth = 8
        self.kernel_size = 41 - 1
        self.bottleneck_size = 32
        self.lr = 0.001

        super().__init__(input_shape, nb_classes)

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > self.bottleneck_size:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape[::-1])

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)

            if self.use_residual and d % 4 == 3:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(self.lr),
                      metrics=[keras.metrics.Recall()])

        return model

    def prepare(self, x_train, y_train, x_test, y_test):
        self.batch_size = int(min(x_train.shape[0] / 10, self.batch_size))
        return x_train.swapaxes(1, 2), y_train, x_test.swapaxes(1, 2), y_test


class Model_INCEPTION_CONV_INSTEAD_POOLING(BaseModel):
    def __init__(self, input_shape, nb_classes):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        self.callbacks = [reduce_lr]

        self.batch_size = 64
        # self.nb_epochs = 2
        self.nb_epochs = 800

        self.nb_filters = 32
        self.use_residual = True
        self.use_bottleneck = True
        self.depth = 6
        self.kernel_size = 41 - 1
        self.bottleneck_size = 32
        self.lr = 0.001

        super().__init__(input_shape, nb_classes)

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > self.bottleneck_size:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        # max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)
        max_pool_1 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=3, padding='same',
                                         activation='relu')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape[::-1])

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(self.lr),
                      metrics=[keras.metrics.Recall()])

        return model

    def prepare(self, x_train, y_train, x_test, y_test):
        self.batch_size = int(min(x_train.shape[0] / 10, self.batch_size))
        return x_train.swapaxes(1, 2), y_train, x_test.swapaxes(1, 2), y_test


class Model_INCEPTION_CONV_INSTEAD_GAP(BaseModel):
    def __init__(self, input_shape, nb_classes):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        self.callbacks = [reduce_lr]

        self.batch_size = 64
        # self.nb_epochs = 2
        self.nb_epochs = 800

        self.nb_filters = 32
        self.use_residual = True
        self.use_bottleneck = True
        self.depth = 6
        self.kernel_size = 41 - 1
        self.bottleneck_size = 32
        self.lr = 0.001

        super().__init__(input_shape, nb_classes)

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > self.bottleneck_size:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape[::-1])

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=input_shape[1],
                                        activation='relu')(x)
        gap_layer = keras.backend.squeeze(gap_layer, 1)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(self.lr),
                      metrics=[keras.metrics.Recall()])

        return model

    def prepare(self, x_train, y_train, x_test, y_test):
        self.batch_size = int(min(x_train.shape[0] / 10, self.batch_size))
        return x_train.swapaxes(1, 2), y_train, x_test.swapaxes(1, 2), y_test


class ZolotyhNet_CONV_INSTEAD_POOLING(nn.Module):
    def __init__(self, input_shape, num_classes=8):
        super().__init__()

        channels = input_shape[1]
        self.features_up = nn.Sequential(
            nn.Conv1d(channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 8, kernel_size=2, stride=2),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            Flatten(),
        )

        self.features_down = nn.Sequential(
            Flatten(),
            nn.Linear(input_shape[0] * channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, input_shape[0]//16),
        )

        self.classifier = nn.Linear(input_shape[0]//16, num_classes)

    def forward(self, x):
        out_up = self.features_up(x)
        out_down = self.features_down(x)
        out_middle = out_up + out_down

        out = self.classifier(out_middle)

        return out


class Model_ZolotyhNet_CONV_INSTEAD_POOLING(BaseModel):
    def __init__(self, input_shape, nb_classes):
        self.batch_size = 128
        # self.batch_size = 16
        self.nb_epochs = 1000
        # self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = ZolotyhNet_CONV_INSTEAD_POOLING(input_shape[::-1], num_classes=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class ZolotyhNet_EXTRA_LAYER(nn.Module):
    def __init__(self, input_shape, num_classes=8):
        super().__init__()

        channels = input_shape[1]
        self.features_up = nn.Sequential(
            nn.Conv1d(channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
            Flatten(),
        )

        self.features_down = nn.Sequential(
            Flatten(),
            nn.Linear(input_shape[0] * channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, input_shape[0]//32),
        )

        self.classifier = nn.Linear(input_shape[0]//32, num_classes)

    def forward(self, x):
        out_up = self.features_up(x)
        out_down = self.features_down(x)
        out_middle = out_up + out_down

        out = self.classifier(out_middle)

        return out


class Model_ZolotyhNet_EXTRA_LAYER(BaseModel):
    def __init__(self, input_shape, nb_classes):
        self.batch_size = 128
        # self.batch_size = 16
        self.nb_epochs = 1000
        # self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = ZolotyhNet_EXTRA_LAYER(input_shape[::-1], num_classes=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class ZolotyhNet_EXTRA_SUBNET(nn.Module):
    def __init__(self, input_shape, num_classes=8):
        super().__init__()

        channels = input_shape[1]
        self.features_up = nn.Sequential(
            nn.Conv1d(channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            Flatten(),
        )

        self.features_down = nn.Sequential(
            Flatten(),
            nn.Linear(input_shape[0] * channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, input_shape[0]//16),
        )

        self.features_mix = nn.Sequential(
            nn.Conv1d(channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            Flatten(),
            nn.Linear(input_shape[0] // 4, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, input_shape[0]//16),
        )

        self.classifier = nn.Linear(input_shape[0]//16, num_classes)

    def forward(self, x):
        out_up = self.features_up(x)
        out_down = self.features_down(x)
        out_mix = self.features_mix(x)
        out_middle = out_up + out_down + out_mix

        out = self.classifier(out_middle)

        return out


class Model_ZolotyhNet_EXTRA_SUBNET(BaseModel):
    def __init__(self, input_shape, nb_classes):
        self.batch_size = 128
        # self.batch_size = 16
        self.nb_epochs = 1000
        # self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = ZolotyhNet_EXTRA_SUBNET(input_shape[::-1], num_classes=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class HeartNet2D_EXTRA_LAYER(nn.Module):
    def __init__(self, input_shape, num_classes=7):
        super(HeartNet2D_EXTRA_LAYER, self).__init__()

        h, w = input_shape
        self.result_size = ((h-1) // 16 + 1) * ((w-1) // 16 + 1)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(64, eps=0.001),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(64, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128, eps=0.001),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(256, eps=0.001),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(256, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(512, eps=0.001),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(512, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.result_size * 512, 2048),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2048, eps=0.001),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.result_size * 512)
        x = self.classifier(x)
        return x


class Model_HeartNet2D_EXTRA_LAYER(BaseModel):
    def __init__(self, input_shape, nb_classes):
        self.batch_size = 128
        # self.batch_size = 16
        self.nb_epochs = 300
        # self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = HeartNet2D_EXTRA_LAYER(input_shape[::-1], num_classes=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model

    def prepare(self, x_train, y_train, x_test, y_test):
        x_train = np.expand_dims(x_train, 1)
        x_test = np.expand_dims(x_test, 1)
        return x_train, y_train, x_test, y_test
