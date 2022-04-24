from enum import Enum

from src.models import dl_4_tsc, LSTMs, lxdv
from src.models.base_model import BaseModel


class Models(Enum):
    FCN = 'fcn'
    MLP = 'mlp'
    RESNET = 'resnet'
    TLENET = 'tlenet'
    MCNN = 'mcnn'
    TWIESN = 'twiesn'
    ENCODER = 'encoder'
    MCDCNN = 'mcdcnn'
    CNN = 'cnn'
    INCEPTION = 'inception'
    LSTMS = 'lstms'
    EcgResNet34 = 'EcgResNet34'
    ZolotyhNet = 'ZolotyhNet'
    HeartNet1D = 'HeartNet1D'
    HeartNet2D = 'HeartNet2D'


model_names_to_models = {
    Models.FCN: dl_4_tsc.Model_FCN,
    Models.MLP: dl_4_tsc.Model_MLP,
    Models.RESNET: dl_4_tsc.Model_RESNET,
    Models.TLENET: dl_4_tsc.Model_TLENET,
    # Models.MCNN: Model_MCNN,
    # Models.TWIESN: Model_TWIESN,
    Models.ENCODER: dl_4_tsc.Model_ENCODER,
    Models.MCDCNN: dl_4_tsc.Model_MCDCNN,
    Models.CNN: dl_4_tsc.Model_CNN,
    Models.INCEPTION: dl_4_tsc.Model_INCEPTION,
    Models.LSTMS: LSTMs.Model_LSTMS,
    Models.EcgResNet34: lxdv.Model_EcgResNet34,
    Models.ZolotyhNet: lxdv.Model_ZolotyhNet,
    Models.HeartNet1D: lxdv.Model_HeartNet1D,
    Models.HeartNet2D: lxdv.Model_HeartNet2D,
}


def get_model_class(model: Models) -> [BaseModel]:
    return model_names_to_models[model]
