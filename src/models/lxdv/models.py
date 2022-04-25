import numpy as np
import torch

from src.models.base_model import BaseModel
from src.models.lit_model import LitModule
from src.models.lxdv.author_models1d import EcgResNet34, ZolotyhNet, HeartNet1D
from src.models.lxdv.author_models2d import HeartNet2D


class Model_EcgResNet34(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = EcgResNet34(input_shape[::-1], num_classes=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_ZolotyhNet(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = ZolotyhNet(input_shape[::-1], num_classes=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_HeartNet1D(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = HeartNet1D(input_shape[::-1], num_classes=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_HeartNet2D(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = HeartNet2D(input_shape[::-1], num_classes=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model

    def prepare(self, x_train, y_train, x_test, y_test):
        x_train = np.expand_dims(x_train, 1)
        x_test = np.expand_dims(x_test, 1)
        return x_train, y_train, x_test, y_test
