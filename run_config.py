from src.constants import Archives
from src.models.model_mapping import Models


DATASETS_TO_RUN = {}
DATASETS_TO_RUN[Archives.UCR_2018] = [
    # 'CinCECGTorso',
    'ECG200',
    # 'ECG5000',
    # 'ECGFiveDays',
    # 'NonInvasiveFetalECGThorax1',
    # 'NonInvasiveFetalECGThorax2',
    # 'TwoLeadECG',
]
# DATASETS_TO_RUN[Archives.PTB] = ['ptb-diagnostic-ecg-database-1.0.0']
DATASETS_TO_RUN[Archives.UEA_2018] = [
    'AtrialFibrillation',
    # 'StandWalkJump'
]

MODELS_TO_RUN = [
    # Models.MLP,
    # Models.CNN,
    # Models.ENCODER,
    # Models.FCN,
    # Models.RESNET,
    # Models.INCEPTION,
    # Models.TLENET,
    # Models.MCDCNN,
    # Models.LSTMS,
    # Models.EcgResNet34,
    # Models.ZolotyhNet,
    # Models.HeartNet1D,
    # Models.HeartNet2D,
]

PRINT_METRICS = True
PRINT_READING = True
SAVE_HISTORY = True
SAVE_MODEL = True
DRAW_GRAPH = True
SHORT_DATA = False
