from src.constants import Archives
from src.models.model_mapping import Models


DATASETS_TO_RUN = {}
DATASETS_TO_RUN[Archives.UCR_2018] = [
    # 'CinCECGTorso',
    'ECG200',
    'ECG5000',
    # 'ECGFiveDays',
    'NonInvasiveFetalECGThorax1',
    'NonInvasiveFetalECGThorax2',
    # 'TwoLeadECG',
]
# DATASETS_TO_RUN[Archives.PTB] = ['ptb-diagnostic-ecg-database-1.0.0']
DATASETS_TO_RUN[Archives.UEA_2018] = [
    'AtrialFibrillation',
    # 'StandWalkJump'
]

MODELS_TO_RUN = [
    # Models.MLP_dl4tsc,
    # Models.CNN,
    # Models.ENCODER,
    # Models.FCN_dl4tsc,
    # Models.RESNET,
    # Models.INCEPTION,
    # Models.TLENET,
    # Models.MCDCNN,
    # Models.LSTMS,
    # Models.EcgResNet34,
    # Models.ZolotyhNet,
    # Models.HeartNet1D,
    # Models.HeartNet2D,

    # Models.FCN,
    # Models.FCNPlus,
    # Models.TCN,
    # Models.InceptionTime,
    # Models.InceptionTimePlus,
    # Models.MLP,
    # Models.gMLP,
    # Models.mWDN,
    # Models.OmniScaleCNN,
    # Models.ResCNN,
    # Models.ResNet,
    # Models.ResNetPlus,
    # Models.RNN,
    # Models.RNNPlus,
    # Models.RNN_FCN,
    # Models.RNN_FCNPlus,
    # Models.TransformerModel,
    # Models.TST,
    # Models.TSTPlus,
    # Models.XceptionTime,
    # Models.XceptionTimePlus,
    # Models.XCM,
    # Models.XCMPlus,
    # Models.XResNet1d,
    # Models.XResNet1dPlus,
    # Models.TSPerceiver,
    # Models.TSiTPlus,

    Models.ENCODER_ORIG,
    Models.ENCODER_NO_POOLING,
    Models.ENCODER_CONV_INSTEAD_POOLING,
    Models.ENCODER_BATCH_NORM,
    Models.ENCODER_EXTRA_LAYER,

    Models.INCEPTION_ORIG,
    Models.INCEPTION_EXTRA_LAYER,
    Models.INCEPTION_BIG_STEP,
    Models.INCEPTION_CONV_INSTEAD_POOLING,
    Models.INCEPTION_CONV_INSTEAD_GAP,

    Models.ZolotyhNet_ORIG,
    Models.ZolotyhNet_CONV_INSTEAD_POOLING,
    Models.ZolotyhNet_EXTRA_LAYER,
    Models.ZolotyhNet_EXTRA_SUBNET,
]

MODELS_FOR_ENSEMBLE = [
    Models.ENCODER,
    Models.INCEPTION,
]

PRINT_METRICS = True
PRINT_READING = True
SAVE_HISTORY = True
SAVE_MODEL = True
DRAW_GRAPH = True
SHORT_DATA = False
PRINT_EPOCH_NUM = True
ENS_NAME = 'ENSEMBLE_f2_orig'
