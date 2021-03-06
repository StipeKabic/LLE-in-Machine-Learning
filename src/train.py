import pandas as pd
from sklearn.model_selection import train_test_split
from enums import Models, Methods, make_grid
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from DatasetLoader import DatasetLoader
from DatasetClass import DatasetClass
from tqdm import tqdm
from lle import LLE
from LLEClass import LLEClass
import random

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, QuantileTransformer

import warnings


class SplitDataset:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


def apply_method(x, method, method_config, LLEClass):
    if method == Methods.PCA:
        pca = PCA(**method_config[method])
        x = pca.fit_transform(x)
    elif method == Methods.Full:
        pass
    elif method == Methods.LLE:
        # TODO: implementirati za LLE slično kao za PCA - primi x (dataframe s featurima) i vrati dataframe s LLE featurima
        x = LLEClass.return_dataframe(method_config[method]["n_components"])
        x = LLEClass.return_dataframe(method_config[method]["n_components"])
        pass
    else:
        raise NotImplementedError
    return x


def get_xy(dataset):
    try:
        dataset.data = dataset.data.sample(1000)
    except:
        pass

    x, y = dataset.data.drop(columns=[dataset.target_name]), dataset.data[dataset.target_name]

    return x, y


def split_dataset(x, y, method, method_config, LLEClass):
    x = apply_method(x, method, method_config, LLEClass)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)  # , random_state=42)
    return SplitDataset(x_train, x_test, y_train, y_test)


class Trainer:
    def __init__(self, classification_model_config, regression_model_config, datasets, method_config, dimension_config):
        """
        Initialize trainer class
        :param classification_model_config: config for classification model hyperparameters
        :param regression_model_config: config for regression model hyperparameters
        :param datasets: list of datasets - dataset is a class with attributes: name(string), data(pandas dataframe), target_name(string, name of target variable)
        :param method_config: config with hyperparameters for dimensionality reduciton methods
        """

        # self.classification_models = {model: self.init_models(model)(**classification_model_config[model]) for model in classification_model_config}
        # self.regression_models = {model: self.init_models(model)(**regression_model_config[model]) for model in regression_model_config}

        self.regression_models = [Models.Linear, Models.RFR]
        self.classification_models = [Models.Logistic, Models.RFC]

        self.models = list(classification_model_config.keys())
        self.config = classification_model_config

        self.datasets = datasets
        self.method_config = method_config
        self.dimensions = dimension_config
        self.results = []

    def init_models(self, model):
        if model == Models.Logistic:
            return LogisticRegression
        elif model == Models.RFC:
            return RandomForestClassifier
        elif model == Models.RFR:
            return RandomForestRegressor
        elif model == Models.Linear:
            return Ridge  # LinearRegression
        else:
            raise NotImplementedError

    def train(self, model_name, _model, dataset, type, LLEClass, x, y):
        scores = {}
        NUM_OF_SPLITS = 5

        # model = make_pipeline(PowerTransformer(), RobustScaler(), _model)
        model = make_pipeline(RobustScaler(), _model)

        "train with original data"
        dimension = len(dataset.data.columns) - 1
        for seed in range(NUM_OF_SPLITS):
            np.random.seed(seed)
            split = split_dataset(x, y, Methods.Full, self.method_config, LLEClass)
            model.fit(split.x_train, split.y_train)
            score = model.score(split.x_test, split.y_test) / NUM_OF_SPLITS
            if seed == 0:
                scores[Methods.Full.value + f"_{dimension}"] = score
            else:
                scores[Methods.Full.value + f"_{dimension}"] += score

        for method in self.method_config:
            # print(f"Training {model_name} on {dataset.name} with {method.value}")
            for dimension in self.dimensions[dataset.name]:
                for seed in range(NUM_OF_SPLITS):
                    np.random.seed(seed)
                    self.method_config[method]["n_components"] = dimension
                    split = split_dataset(x, y, method, self.method_config, LLEClass)

                    model.fit(split.x_train, split.y_train)
                    score = model.score(split.x_test, split.y_test) / NUM_OF_SPLITS
                    if seed == 0:
                        scores[method.value + f"_{dimension}"] = score
                    else:
                        scores[method.value + f"_{dimension}"] += score

        return scores

    def accumulate_results(self, old_results, new_results):
        if len(old_results.keys()) == 0:
            old_results = new_results
        else:
            old_results = {k: max(old_results[k], new_results[k]) for k in new_results.keys()}
        return old_results

    def train_all_combinations(self):
        for dataset in self.datasets:
            x, y = get_xy(dataset)
            LLE = LLEClass(**self.method_config[Methods.LLE],
                           X=x)
            if dataset.type == "r":
                models = self.regression_models
            elif dataset.type == "c":
                models = self.classification_models
            else:
                models = self.classification_models

            for model_name in models:
                model_class = self.init_models(model_name)
                grid = make_grid(self.config[model_name])
                results = {}
                for params in tqdm(grid):
                    if model_name != Models.Linear:
                        params["n_jobs"] = -1
                    model = model_class(**params)

                    new_results = self.train(model_name, model, dataset, dataset.type, LLE, x, y)
                    results = self.accumulate_results(results, new_results)

                results["model"] = model_name.value
                results["dataset"] = dataset.name
                self.results.append(results)

    def process_results(self):
        results = pd.DataFrame(self.results)
        results = results.melt(id_vars=["model", "dataset"])
        results["method"] = results["variable"].apply(lambda x: x.split("_")[0])
        results["dimension"] = results["variable"].apply(lambda x: x.split("_")[1])
        results = results.drop(columns=["variable"])
        results = results.rename(columns={"value": "score"})
        results = results.dropna().reset_index(drop=True)
        return results


def init_datasets():
    loader = DatasetLoader()
    datasets = []
    for name in loader.names:
        ds = loader.datasets[name]
        ds["name"] = name
        datasets.append(DatasetClass(**ds))

    return datasets


def warn(*args, **kwargs):
    pass


def main():
    # warnings.warn = warn

    random.seed(42)

    model_config_file = {
        Models.Logistic: {
            "penalty": ['l2'],
            "C": [0.01, 0.2, 0.5, 0.7, 0.9, 1, 1.2, 1.5],
            "max_iter": [300]
        },
        Models.Linear: {
            "alpha": [0.5, 0.7, 1, 1.5, 2],
            "max_iter": [300]
        },
        Models.RFC: {"n_estimators": [150],
                     "max_depth": [5],
                     "min_samples_split": [2, 4, 6],
                     "min_samples_leaf": [2, 4],
                     "max_features": ["sqrt"]}
    }

    model_config_file[Models.RFR] = model_config_file[Models.RFC]

    # TODO: add Methods.LLE config
    method_config_file = {Methods.PCA: {"n_components": None},
                          Methods.LLE: {"n_components": None, "r": 0.001, "k": 35}}

    datasets = init_datasets()

    dimension_config_file = {dataset.name: list(range(1, len(dataset.data.columns) - 1)) for dataset in datasets}

    # trainer = Trainer(classification_model_config_file, regression_model_config_file, datasets, method_config_file, dimension_config_file)
    trainer = Trainer(model_config_file, model_config_file, datasets, method_config_file, dimension_config_file)
    trainer.train_all_combinations()

    results = trainer.process_results()
    print(results)
    print(results[(results.model == "Linear Regression") \
                  & (results.dataset == "ames_housing")][["method", "score", "dimension"]])
    results.to_csv("../data/results.csv")


if __name__ == "__main__":
    main()
