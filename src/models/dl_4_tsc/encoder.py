# hfawaz/dl-4-tsc proposed model CNN + LSTM
import tensorflow_addons as tfa
from tensorflow import keras

from src.models.base_model import BaseModel


class Model_ENCODER(BaseModel):
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
