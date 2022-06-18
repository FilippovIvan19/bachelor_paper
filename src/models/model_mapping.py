from enum import Enum

from src.models import dl_4_tsc, LSTMs, lxdv, tsai, experiments
from src.models.base_model import BaseModel


class Models(Enum):
    FCN_dl4tsc = 'fcn_dl4tsc'
    MLP_dl4tsc = 'mlp_dl4tsc'
    RESNET = 'resnet_dl4tsc'
    TLENET = 'tlenet'
    ENCODER = 'encoder'
    MCDCNN = 'mcdcnn'
    CNN = 'cnn'
    INCEPTION = 'inception'
    LSTMS = 'lstms'
    EcgResNet34 = 'EcgResNet34'
    ZolotyhNet = 'ZolotyhNet'
    HeartNet1D = 'HeartNet1D'
    HeartNet2D = 'HeartNet2D'

    FCN = 'FCN'
    FCNPlus = 'FCNPlus'
    TCN = 'TCN'
    InceptionTime = 'InceptionTime'
    InceptionTimePlus = 'InceptionTimePlus'
    MLP = 'MLP'
    gMLP = 'gMLP'
    mWDN = 'mWDN'
    OmniScaleCNN = 'OmniScaleCNN'
    ResCNN = 'ResCNN'
    ResNet = 'ResNet'
    ResNetPlus = 'ResNetPlus'
    RNN = 'RNN'
    RNNPlus = 'RNNPlus'
    RNN_FCN = 'RNN_FCN'
    RNN_FCNPlus = 'RNN_FCNPlus'
    TransformerModel = 'TransformerModel'
    TST = 'TST'
    TSTPlus = 'TSTPlus'
    XceptionTime = 'XceptionTime'
    XceptionTimePlus = 'XceptionTimePlus'
    XCM = 'XCM'
    XCMPlus = 'XCMPlus'
    XResNet1d = 'XResNet1d'
    XResNet1dPlus = 'XResNet1dPlus'
    TSPerceiver = 'TSPerceiver'
    TSiTPlus = 'TSiTPlus'

    ENCODER_ORIG = 'ENCODER_ORIG'
    ENCODER_NO_POOLING = 'ENCODER_NO_POOLING'
    ENCODER_CONV_INSTEAD_POOLING = 'ENCODER_CONV_INSTEAD_POOLING'
    ENCODER_BATCH_NORM = 'ENCODER_BATCH_NORM'
    ENCODER_EXTRA_LAYER = 'ENCODER_EXTRA_LAYER'

    INCEPTION_ORIG = 'INCEPTION_ORIG'
    INCEPTION_EXTRA_LAYER = 'INCEPTION_EXTRA_LAYER'
    INCEPTION_BIG_STEP = 'INCEPTION_BIG_STEP'
    INCEPTION_CONV_INSTEAD_POOLING = 'INCEPTION_CONV_INSTEAD_POOLING'
    INCEPTION_CONV_INSTEAD_GAP = 'INCEPTION_CONV_INSTEAD_GAP'

    ZolotyhNet_ORIG = 'ZolotyhNet_ORIG'
    ZolotyhNet_CONV_INSTEAD_POOLING = 'ZolotyhNet_CONV_INSTEAD_POOLING'
    ZolotyhNet_EXTRA_LAYER = 'ZolotyhNet_EXTRA_LAYER'
    ZolotyhNet_EXTRA_SUBNET = 'ZolotyhNet_EXTRA_SUBNET'

    HeartNet2D_ORIG = 'HeartNet2D_ORIG'
    HeartNet2D_EXTRA_LAYER = 'HeartNet2D_EXTRA_LAYER'


model_names_to_models = {
    Models.FCN_dl4tsc: dl_4_tsc.Model_FCN_dl4tsc,
    Models.MLP_dl4tsc: dl_4_tsc.Model_MLP_dl4tsc,
    Models.RESNET: dl_4_tsc.Model_RESNET,
    Models.TLENET: dl_4_tsc.Model_TLENET,
    Models.ENCODER: dl_4_tsc.Model_ENCODER,
    Models.MCDCNN: dl_4_tsc.Model_MCDCNN,
    Models.CNN: dl_4_tsc.Model_CNN,
    Models.INCEPTION: dl_4_tsc.Model_INCEPTION,
    Models.LSTMS: LSTMs.Model_LSTMS,
    Models.EcgResNet34: lxdv.Model_EcgResNet34,
    Models.ZolotyhNet: lxdv.Model_ZolotyhNet,
    Models.HeartNet1D: lxdv.Model_HeartNet1D,
    Models.HeartNet2D: lxdv.Model_HeartNet2D,

    Models.FCN: tsai.Model_FCN,
    Models.FCNPlus: tsai.Model_FCNPlus,
    Models.TCN: tsai.Model_TCN,
    Models.InceptionTime: tsai.Model_InceptionTime,
    Models.InceptionTimePlus: tsai.Model_InceptionTimePlus,
    Models.MLP: tsai.Model_MLP,
    Models.gMLP: tsai.Model_gMLP,
    Models.mWDN: tsai.Model_mWDN,
    Models.OmniScaleCNN: tsai.Model_OmniScaleCNN,
    Models.ResCNN: tsai.Model_ResCNN,
    Models.ResNet: tsai.Model_ResNet,
    Models.ResNetPlus: tsai.Model_ResNetPlus,
    Models.RNN: tsai.Model_RNN,
    Models.RNNPlus: tsai.Model_RNNPlus,
    Models.RNN_FCN: tsai.Model_RNN_FCN,
    Models.RNN_FCNPlus: tsai.Model_RNN_FCNPlus,
    Models.TransformerModel: tsai.Model_TransformerModel,
    Models.TST: tsai.Model_TST,
    Models.TSTPlus: tsai.Model_TSTPlus,
    Models.XceptionTime: tsai.Model_XceptionTime,
    Models.XceptionTimePlus: tsai.Model_XceptionTimePlus,
    Models.XCM: tsai.Model_XCM,
    Models.XCMPlus: tsai.Model_XCMPlus,
    Models.XResNet1d: tsai.Model_XResNet1d,
    Models.XResNet1dPlus: tsai.Model_XResNet1dPlus,
    Models.TSPerceiver: tsai.Model_TSPerceiver,
    Models.TSiTPlus: tsai.Model_TSiTPlus,

    Models.ENCODER_ORIG: experiments.Model_ENCODER_ORIG,
    Models.ENCODER_NO_POOLING: experiments.Model_ENCODER_NO_POOLING,
    Models.ENCODER_CONV_INSTEAD_POOLING: experiments.Model_ENCODER_CONV_INSTEAD_POOLING,
    Models.ENCODER_BATCH_NORM: experiments.Model_ENCODER_BATCH_NORM,
    Models.ENCODER_EXTRA_LAYER: experiments.Model_ENCODER_EXTRA_LAYER,

    Models.INCEPTION_ORIG: experiments.Model_INCEPTION_ORIG,
    Models.INCEPTION_EXTRA_LAYER: experiments.Model_INCEPTION_EXTRA_LAYER,
    Models.INCEPTION_BIG_STEP: experiments.Model_INCEPTION_BIG_STEP,
    Models.INCEPTION_CONV_INSTEAD_POOLING: experiments.Model_INCEPTION_CONV_INSTEAD_POOLING,
    Models.INCEPTION_CONV_INSTEAD_GAP: experiments.Model_INCEPTION_CONV_INSTEAD_GAP,

    Models.ZolotyhNet_ORIG: experiments.Model_ZolotyhNet_ORIG,
    Models.ZolotyhNet_CONV_INSTEAD_POOLING: experiments.Model_ZolotyhNet_CONV_INSTEAD_POOLING,
    Models.ZolotyhNet_EXTRA_LAYER: experiments.Model_ZolotyhNet_EXTRA_LAYER,
    Models.ZolotyhNet_EXTRA_SUBNET: experiments.Model_ZolotyhNet_EXTRA_SUBNET,

    Models.HeartNet2D_ORIG: experiments.Model_HeartNet2D_ORIG,
    Models.HeartNet2D_EXTRA_LAYER: experiments.Model_HeartNet2D_EXTRA_LAYER,

}


def get_model_class(model: Models) -> [BaseModel]:
    return model_names_to_models[model]
