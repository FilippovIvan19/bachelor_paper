import abc
import os
import shutil
import time
from typing import Type
import pandas as pd
import torch
from pytorch_lightning import Trainer
from tensorflow import keras
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint

from src.constants import HISTORY_COLUMN_NAMES
from src.models.base_model import BaseModel
from src.models.dl_4_tsc import Model_TLENET
from src.models.lit_model import LitModule
from src.models.model_mapping import get_model_class, Models
from src.utils import calculate_metrics, draw_history_graph, draw_confusion_matrix


class BaseClassifier(abc.ABC):
    def __init__(self, data, model: Models, history_dir, model_file='default_model_name.txt', print_epoch_num=True):
        self.history_dir = history_dir
        self.model_path = history_dir + model_file
        self.print_epoch_num = print_epoch_num

        self.x_train, self.y_train, self.x_test, self.y_test, self.y_test_labeled, self.encoder, \
            input_shape, nb_classes = data

        model_class: [BaseModel] = get_model_class(model)
        self.model = model_class(input_shape, nb_classes)

    def run(self, *, save_history=False, save_model=False, draw_graph=False):
        if os.path.exists(self.history_dir):
            shutil.rmtree(self.history_dir)
        os.makedirs(self.history_dir)

        self.x_train, self.y_train, self.x_test, self.y_test = \
            self.model.prepare(self.x_train, self.y_train, self.x_test, self.y_test)
        train_start_time = time.time()
        history = self.fit()
        train_duration = time.time() - train_start_time

        if save_history:
            hist_df = pd.DataFrame(history, columns=HISTORY_COLUMN_NAMES)
            hist_df.to_csv(self.history_dir + 'history.csv')

        if draw_graph:
            draw_history_graph(history, self.history_dir)

        y_predicted = self.predict(self.x_test)
        y_predicted_labeled = self.encoder.inverse_transform(y_predicted)

        if not save_model:
            if os.path.exists(self.model_path):
                os.remove(self.model_path)

        precision, accuracy, recall = calculate_metrics(self.y_test_labeled, y_predicted_labeled)
        return [precision, accuracy, recall, train_duration/60]

    def run_train_only(self, *, save_history=False, draw_graph=False):
        if os.path.exists(self.history_dir):
            shutil.rmtree(self.history_dir)
        os.makedirs(self.history_dir)

        self.x_train, self.y_train, self.x_test, self.y_test = \
            self.model.prepare(self.x_train, self.y_train, self.x_test, self.y_test)
        train_start_time = time.time()
        history = self.fit()
        train_duration = time.time() - train_start_time

        if save_history:
            hist_df = pd.DataFrame(history, columns=HISTORY_COLUMN_NAMES)
            hist_df.to_csv(self.history_dir + 'history.csv')

        if draw_graph:
            draw_history_graph(history, self.history_dir)

        return train_duration/60

    def run_eval_only(self, model_name):
        self.x_train, self.y_train, self.x_test, self.y_test = \
            self.model.prepare(self.x_train, self.y_train, self.x_test, self.y_test)

        y_predicted = self.predict(self.x_test)
        y_predicted_labeled = self.encoder.inverse_transform(y_predicted)

        draw_confusion_matrix(self.y_test_labeled, y_predicted_labeled, self.history_dir, model_name)

        precision, accuracy, recall = calculate_metrics(self.y_test_labeled, y_predicted_labeled)
        return [precision, accuracy, recall, 0.]

    @abc.abstractmethod
    def fit(self):
        return

    def predict(self, x):
        return self.model.predict(x)

    def predict_probabilities(self):
        self.x_train, self.y_train, self.x_test, self.y_test = \
            self.model.prepare(self.x_train, self.y_train, self.x_test, self.y_test)
        return self.predict(self.x_test)


class ClassifierKeras(BaseClassifier):
    def __init__(self, train_test_data, model: Models, history_dir, print_epoch_num):
        super().__init__(train_test_data, model, history_dir, 'best_model.hdf5', print_epoch_num)

    def fit(self):
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self.model_path, save_best_only=True
        )

        verbose = 2 if self.print_epoch_num else 0
        history = self.model.model.fit(
            self.x_train, self.y_train, batch_size=self.model.batch_size, epochs=self.model.nb_epochs,
            validation_data=(self.x_test, self.y_test), callbacks=[model_checkpoint] + self.model.callbacks,
            verbose=verbose
        )

        keras.backend.clear_session()
        return history.history

    def predict(self, x):
        if isinstance(self.model, Model_TLENET):
            y_predicted = self.model.tlenet_predict(x, self.model_path)
            y_predicted = self.encoder.categories_[0][y_predicted.astype(int)]  # int?
            return self.encoder.transform(y_predicted.reshape(-1, 1)).toarray()

        model = keras.models.load_model(self.model_path)
        return model.predict(x)


class ClassifierTorch(BaseClassifier):
    def __init__(self, train_test_data, model: Models, history_dir, print_epoch_num):
        self.best_model_file_name = 'best_model'
        super().__init__(train_test_data, model, history_dir, self.best_model_file_name + '.ckpt', print_epoch_num)

    def fit(self):
        train_dataset = TensorDataset(torch.from_numpy(self.x_train).float(), torch.from_numpy(self.y_train).float())
        test_dataset = TensorDataset(torch.from_numpy(self.x_test).float(), torch.from_numpy(self.y_test).float())
        train_dataloader = DataLoader(train_dataset, batch_size=self.model.batch_size, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=self.model.batch_size, num_workers=0)

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.history_dir, filename=self.best_model_file_name, monitor='val_loss')
        trainer = Trainer(callbacks=[checkpoint_callback], max_epochs=self.model.nb_epochs,
                          enable_progress_bar=self.print_epoch_num, enable_model_summary=False, accelerator='auto')

        trainer.fit(self.model.model, train_dataloader, test_dataloader)
        return self.model.model.history

    def predict(self, x):
        test_dataset = torch.from_numpy(x).float()
        test_dataloader = DataLoader(test_dataset, batch_size=self.model.batch_size, num_workers=0)

        model = LitModule.load_from_checkpoint(self.model_path,
                                               model=self.model.model.model, optimizer=self.model.model.optimizer)
        trainer = Trainer(enable_progress_bar=self.print_epoch_num, accelerator='auto')
        y_pred_list = trainer.predict(model, dataloaders=test_dataloader)
        return torch.cat(y_pred_list, 0)

    def predict_probabilities(self):
        return F.softmax(super().predict_probabilities()).numpy()


classifier_to_model_names = {
    ClassifierKeras: [
        Models.FCN_dl4tsc, Models.MLP_dl4tsc, Models.RESNET, Models.TLENET,
        Models.ENCODER, Models.MCDCNN, Models.CNN, Models.INCEPTION,
        Models.LSTMS,

        Models.ENCODER_ORIG,
        Models.ENCODER_NO_POOLING, Models.ENCODER_CONV_INSTEAD_POOLING,
        Models.ENCODER_BATCH_NORM, Models.ENCODER_EXTRA_LAYER,

        Models.INCEPTION_ORIG,
        Models.INCEPTION_EXTRA_LAYER, Models.INCEPTION_BIG_STEP,
        Models.INCEPTION_CONV_INSTEAD_POOLING, Models.INCEPTION_CONV_INSTEAD_GAP,
    ],
    ClassifierTorch: [
        Models.EcgResNet34, Models.ZolotyhNet, Models.HeartNet1D, Models.HeartNet2D,

        Models.FCN,
        Models.FCNPlus,
        Models.TCN,
        Models.InceptionTime,
        Models.InceptionTimePlus,
        Models.MLP,
        Models.gMLP,
        Models.mWDN,
        Models.OmniScaleCNN,
        Models.ResCNN,
        Models.ResNet,
        Models.ResNetPlus,
        Models.RNN,
        Models.RNNPlus,
        Models.RNN_FCN,
        Models.RNN_FCNPlus,

        Models.TransformerModel,
        Models.TST,
        Models.TSTPlus,
        Models.XceptionTime,
        Models.XceptionTimePlus,
        Models.XCM,
        Models.XCMPlus,
        Models.XResNet1d,
        Models.XResNet1dPlus,
        Models.TSPerceiver,
        Models.TSiTPlus,

        Models.ZolotyhNet_ORIG,
        Models.ZolotyhNet_CONV_INSTEAD_POOLING,
        Models.ZolotyhNet_EXTRA_LAYER,
        Models.ZolotyhNet_EXTRA_SUBNET,

        Models.HeartNet2D_ORIG,
        Models.HeartNet2D_EXTRA_LAYER,
    ]
}


def get_classifier_class(model: Models) -> Type[BaseClassifier]:
    for item in classifier_to_model_names.items():
        if model in item[1]:
            return item[0]
    raise ValueError("no classifier found for model {}".format(model.value))
