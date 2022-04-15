import abc
import os
import shutil
import time
from typing import Type
import pandas as pd
from tensorflow import keras

from src.models.base_model import BaseModel
from src.models.model_mapping import get_model_class, Models
from src.utils import calculate_metrics, draw_history_graph


class BaseClassifier(abc.ABC):
    def __init__(self, data, model: Models, history_dir, model_file='default_model_name.txt'):
        self.history_dir = history_dir
        self.model_path = history_dir + model_file

        self.x_train, self.y_train, self.x_test, self.y_test, self.y_test_labeled, self.encoder, \
            input_shape, nb_classes = data

        model_class: [BaseModel] = get_model_class(model)
        self.model = model_class(input_shape, nb_classes)

    def run(self, *, save_history=False, save_model=False, draw_graph=False):
        if os.path.exists(self.history_dir):
            shutil.rmtree(self.history_dir)
        os.makedirs(self.history_dir)

        self.model.prepare(self.x_train, self.y_train, self.x_test, self.y_test)
        train_start_time = time.time()
        history = self.fit()
        train_duration = time.time() - train_start_time

        y_predicted = self.predict(self.x_test)
        y_predicted_labeled = self.encoder.inverse_transform(y_predicted)

        if save_history:
            hist_df = pd.DataFrame(history.history)
            hist_df.to_csv(self.history_dir + 'history.csv')

        if draw_graph:
            draw_history_graph(history, self.history_dir)

        if not save_model:
            if os.path.exists(self.model_path):
                os.remove(self.model_path)

        precision, accuracy, recall = calculate_metrics(self.y_test_labeled, y_predicted_labeled)
        return [precision, accuracy, recall, train_duration/60]

    @abc.abstractmethod
    def fit(self):
        return

    def predict(self, x):
        return self.model.predict(x)


class Classifier_DL4TSC(BaseClassifier):
    def __init__(self, train_test_data, model: Models, history_dir):
        super().__init__(train_test_data, model, history_dir, 'best_model.hdf5')

    def fit(self):
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self.model_path, save_best_only=True
        )

        history = self.model.model.fit(
            self.x_train, self.y_train, batch_size=self.model.batch_size, epochs=self.model.nb_epochs,
            validation_data=(self.x_test, self.y_test), callbacks=[model_checkpoint] + self.model.callbacks, verbose=0
        )

        keras.backend.clear_session()
        return history

    def predict(self, x):
        model = keras.models.load_model(self.model_path)
        return model.predict(x)


classifier_to_model_names = {
    Classifier_DL4TSC: [
        Models.FCN, Models.MLP, Models.RESNET, Models.TLENET, Models.MCNN,
        Models.TWIESN, Models.ENCODER, Models.MCDCNN, Models.CNN, Models.INCEPTION
    ]
}


def get_classifier_class(model: Models) -> Type[BaseClassifier]:
    for item in classifier_to_model_names.items():
        if model in item[1]:
            return item[0]
    raise ValueError("no classifier found for model {}".format(model.value))