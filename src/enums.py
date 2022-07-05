from enum import Enum


class Models(str, Enum):
    Logistic = "Logistic Regression"
    RFC = "Random Forest Classifier"
    RFR = "Random Forest Regressor"
    Linear = "Linear Regression"


class Methods(str, Enum):
    Full = "Original dimensionality"
    PCA = "Principal Component Analysis"
    LLE = "Locally linear embedding"
