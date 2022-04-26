import torch

from tsai.models.FCN import FCN
from tsai.models.FCNPlus import FCNPlus
from tsai.models.TCN import TCN
from tsai.models.InceptionTime import InceptionTime
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.models.MLP import MLP
from tsai.models.gMLP import gMLP
from tsai.models.mWDN import mWDN
from tsai.models.OmniScaleCNN import OmniScaleCNN
from tsai.models.ResCNN import ResCNN
from tsai.models.ResNet import ResNet
from tsai.models.ResNetPlus import ResNetPlus
from tsai.models.RNN import RNN
from tsai.models.RNNPlus import RNNPlus
from tsai.models.RNN_FCN import RNN_FCN
from tsai.models.RNN_FCNPlus import RNN_FCNPlus
from tsai.models.TransformerModel import TransformerModel
from tsai.models.TST import TST
from tsai.models.TSTPlus import TSTPlus
from tsai.models.XceptionTime import XceptionTime
from tsai.models.XceptionTimePlus import XceptionTimePlus
from tsai.models.XCM import XCM
from tsai.models.XCMPlus import XCMPlus
from tsai.models.XResNet1d import xresnet1d18
from tsai.models.XResNet1dPlus import xresnet1d18plus
from tsai.models.TSPerceiver import TSPerceiver
from tsai.models.TSiTPlus import TSiTPlus

from src.models.base_model import BaseModel
from src.models.lit_model import LitModule


class Model_FCN(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = FCN(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_FCNPlus(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = FCNPlus(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_TCN(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = TCN(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_InceptionTime(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = InceptionTime(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_InceptionTimePlus(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = InceptionTimePlus(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_MLP(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = MLP(c_in=input_shape[0], c_out=nb_classes, seq_len=input_shape[1])
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_gMLP(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = gMLP(c_in=input_shape[0], c_out=nb_classes, seq_len=input_shape[1])
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_mWDN(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = mWDN(c_in=input_shape[0], c_out=nb_classes, seq_len=input_shape[1])
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_OmniScaleCNN(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = OmniScaleCNN(c_in=input_shape[0], c_out=nb_classes, seq_len=input_shape[1])
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_ResCNN(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = ResCNN(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_ResNet(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = ResNet(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_ResNetPlus(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = ResNetPlus(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_RNN(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = RNN(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_RNNPlus(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = RNNPlus(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_RNN_FCN(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = RNN_FCN(c_in=input_shape[0], c_out=nb_classes, seq_len=input_shape[1])
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_RNN_FCNPlus(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = RNN_FCNPlus(c_in=input_shape[0], c_out=nb_classes, seq_len=input_shape[1])
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_TransformerModel(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = TransformerModel(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_TST(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = TST(c_in=input_shape[0], c_out=nb_classes, seq_len=input_shape[1])
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_TSTPlus(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = TSTPlus(c_in=input_shape[0], c_out=nb_classes, seq_len=input_shape[1])
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_XceptionTime(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = XceptionTime(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_XceptionTimePlus(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = XceptionTimePlus(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_XCM(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = XCM(c_in=input_shape[0], c_out=nb_classes, seq_len=input_shape[1])
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_XCMPlus(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = XCMPlus(c_in=input_shape[0], c_out=nb_classes, seq_len=input_shape[1])
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_XResNet1d(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = xresnet1d18(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_XResNet1dPlus(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = xresnet1d18plus(c_in=input_shape[0], c_out=nb_classes)
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_TSPerceiver(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = TSPerceiver(c_in=input_shape[0], c_out=nb_classes, seq_len=input_shape[1])
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model


class Model_TSiTPlus(BaseModel):
    def __init__(self, input_shape, nb_classes):
        # self.batch_size = 128
        self.batch_size = 16
        # self.nb_epochs = 1000
        self.nb_epochs = 2
        super().__init__(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        inner_model = TSiTPlus(c_in=input_shape[0], c_out=nb_classes, seq_len=input_shape[1])
        optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
        model = LitModule(inner_model, optimizer)
        return model
