from enum import Enum


class Models(str, Enum):
    Logistic = "Logistic Regression"
    RF = "Random Forest"


class Methods(str, Enum):
    Full = "Original dimensionality"
    PCA = "Principal Component Analysis"
    LLE = "Locally linear embedding"
