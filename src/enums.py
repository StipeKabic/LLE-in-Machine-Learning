from enum import Enum
import itertools

class Models(str, Enum):
    Logistic = "Logistic Regression"
    RFC = "Random Forest Classifier"
    RFR = "Random Forest Regressor"
    Linear = "Linear Regression"


class Methods(str, Enum):
    Full = "Original dimensionality"
    PCA = "Principal Component Analysis"
    LLE = "Locally linear embedding"


class TargetType(str, Enum):
    Regression = "regression"
    Classification = "classification"


def make_grid(pars_dict):
    keys = pars_dict.keys()
    combinations = itertools.product(*pars_dict.values())
    ds = [dict(zip(keys, cc)) for cc in combinations]
    return ds
