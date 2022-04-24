from enum import Enum


RESULTS_DIR_SUFFIX = '/../results/'
ARCHIVES_DIR_SUFFIX = '/../archives/'
HISTORY_DIR_SUFFIX = '/../history/'


COLUMN_NAMES = ['model', 'precision', 'accuracy', 'recall', 'duration (minutes)']

PRINT_METRICS_STRING = '''\
        precision = {:.4f}
        accuracy  = {:.4f}
        recall    = {:.4f}
        train duration = {:.2f} minutes
'''


class Archives(Enum):
    UCR_2018 = 'UCRArchive_2018'        # univariate
    UEA_2018 = 'Multivariate2018_ts'    # multivariate
    PTB = 'ptb'


DATASETS = {}
DATASETS[Archives.UCR_2018] = ['ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
                               'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown',
                               'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
                               'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction',
                               'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                               'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'Earthquakes', 'ECG200',
                               'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'EOGHorizontalSignal',
                               'EOGVerticalSignal', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR',
                               'FiftyWords', 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain',
                               'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2',
                               'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint',
                               'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung',
                               'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
                               'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
                               'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
                               'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
                               'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
                               'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
                               'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
                               'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
                               'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
                               'PLAID', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
                               'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
                               'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2',
                               'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapeletSim', 'ShapesAll',
                               'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
                               'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf',
                               'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
                               'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll',
                               'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ',
                               'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']
DATASETS[Archives.UEA_2018] = ['AtrialFibrillation', 'StandWalkJump']
DATASETS[Archives.PTB] = ['ptb-diagnostic-ecg-database-1.0.0']


def check_archive_contains_dataset(datasets):
    for archives in datasets.items():
        cur_archive = archives[0]
        cur_datasets = archives[1]
        for dataset in cur_datasets:
            if dataset not in DATASETS[cur_archive]:
                raise ValueError("'{}' is not a valid dataset name for {} archive".format(dataset, cur_archive.value))
