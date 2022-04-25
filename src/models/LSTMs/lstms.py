from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import keras
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

from src.models.base_model import BaseModel


class Model_LSTMS(BaseModel):
    def __init__(self, input_shape, nb_classes):
        self.callbacks = []
        # self.batch_size = 512
        self.batch_size = 2
        # self.nb_epochs = 50
        self.nb_epochs = 2

        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        dropout = 0.2
        model = Sequential()
        model.add(LSTM(256, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(64))
        model.add(Dropout(dropout))
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[keras.metrics.Recall()])

        return model

    def prepare(self, x_train, y_train, x_test, y_test):
        x_train = pad_sequences(x_train)
        x_test = pad_sequences(x_test)
        return x_train, y_train, x_test, y_test
