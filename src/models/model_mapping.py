from enum import Enum

from src.models import dl_4_tsc, LSTMs
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
}


def get_model_class(model: Models) -> [BaseModel]:
    return model_names_to_models[model]
