from enum import Enum

from src.models import dl_4_tsc
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


model_names_to_models = {
    # Models.FCN: Model_FCN,
    Models.MLP: dl_4_tsc.Model_MLP,
    # Models.RESNET: Model_RESNET,
    # Models.TLENET: Model_TLENET,
    # Models.MCNN: Model_MCNN,
    # Models.TWIESN: Model_TWIESN,
    # Models.ENCODER: Model_ENCODER,
    # Models.MCDCNN: Model_MCDCNN,
    Models.CNN: dl_4_tsc.Model_CNN,
    # Models.INCEPTION: Model_INCEPTION,
}


def get_model_class(model: Models) -> [BaseModel]:
    return model_names_to_models[model]