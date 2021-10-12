from enum import Enum


class Behavior(Enum):
    NORMAL = "normal"
    NORMAL_V2 = "normal_v2"
    DELAY = "delay"
    DISORDER = "disorder"
    FREEZE = "freeze"
    HOP = "hop"
    MIMIC = "mimic"
    NOISE = "noise"
    REPEAT = "repeat"
    SPOOF = "spoof"


class RaspberryPi(Enum):
    PI3_2GB = "pi3-2gb"
    # black cover
    PI4_2GB_BC = "pi4-2gb-bc"
    # white cover
    PI4_2GB_WC = "pi4-2gb-wc"
    PI4_4GB = "pi4-4gb"


class ModelArchitecture(Enum):
    MLP_MONO_CLASS = "MLP_mono_class"
    MLP_MULTI_CLASS = "MLP_multi_class"
    AUTO_ENCODER = "auto_encoder"


class AdversaryType(Enum):
    RANDOM_WEIGHT = "random_weight"
    EXAGGERATE_TRESHOLD = "exaggerate_threshold"
    UNDERSTATE_TRESHOLD = "understate_threshold"


class AggregationMechanism(Enum):
    FED_AVG = "fed_avg",
    TRIMMED_MEAN = "trimmed_mean"


class Scaler(Enum):
    STANDARD_SCALER = "std"
    MINMAX_SCALER = "minmax"
