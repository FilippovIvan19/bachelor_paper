from src.constants import Archives
from src.models.model_mapping import Models


DATASETS_TO_RUN = {}
DATASETS_TO_RUN[Archives.UCR_2018] = [
    'Coffee',
    'Car',
    # 'CinCECGTorso'
]

MODELS_TO_RUN = [
    # Models.MLP,
    Models.CNN,
]

PRINT_METRICS = True
PRINT_READING = True
SAVE_HISTORY = True
SAVE_MODEL = True
DRAW_GRAPH = True

