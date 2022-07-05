import pandas as pd
from sklearn.model_selection import train_test_split
from enums import Models, Methods
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from DatasetLoader import DatasetLoader
from DatasetClass import DatasetClass
from tqdm import tqdm


class SplitDataset:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


def split_dataset(dataset, method, method_config):
    x, y = dataset.data.drop(columns=[dataset.target_name]), dataset.data[dataset.target_name]
    if method == Methods.PCA:
        pca = PCA(**method_config[method])
        x = pca.fit_transform(x)
    elif method == Methods.Full:
        pass
    elif method == Methods.LLE:
        # TODO: implementirati za LLE slično kao za PCA - primi x (dataframe s featurima) i vrati dataframe s LLE featurima
        pass
    else:
        raise NotImplementedError
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
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
        self.classification_models = {model: self.init_models(model)(**classification_model_config[model]) for model in classification_model_config}
        self.regression_models = {model: self.init_models(model)(**regression_model_config[model]) for model in regression_model_config}

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
            return LinearRegression
        else:
            raise NotImplementedError

    def train(self, model_name, dataset, type):
        scores = {}
        if type == "r":
            model = self.regression_models[model_name]
        elif type == "c":
            model = self.classification_models[model_name]
        else:
            raise NotImplementedError

        "train with original data"
        dimension = len(dataset.data.columns) - 1
        split = split_dataset(dataset, Methods.Full, self.method_config)
        model.fit(split.x_train, split.y_train)
        score = model.score(split.x_test, split.y_test)
        scores[Methods.Full.value + f"_{dimension}"] = score

        for method in self.method_config:
            print(f"Training {model_name} on {dataset.name} with {method.value}")
            for dimension in tqdm(self.dimensions[dataset.name]):
                self.method_config[method]["n_components"] = dimension
                split = split_dataset(dataset, method, self.method_config)
                model.fit(split.x_train, split.y_train)
                score = model.score(split.x_test, split.y_test)
                scores[method.value + f"_{dimension}"] = score

        return scores

    def train_all_combinations(self):
        for dataset in self.datasets:
            if dataset.type == "r":
                models = self.regression_models
            elif dataset.type == "c":
                models = self.classification_models
            else:
                raise NotImplementedError
            for model_name in models:
                results = self.train(model_name, dataset, dataset.type)
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


def main():
    classification_model_config_file = {Models.Logistic: {"penalty": 'none'},
                                        Models.RFC: {"n_estimators": 20,
                                                     "max_depth": 5,
                                                     "min_samples_split": 2,
                                                     "min_samples_leaf": 1,
                                                     "max_features": "sqrt"}
                                        }

    regression_model_config_file = {Models.Linear: {},
                                    Models.RFR: {"n_estimators": 20,
                                                 "max_depth": 5,
                                                 "min_samples_split": 2,
                                                 "min_samples_leaf": 1,
                                                 "max_features": "sqrt"}
                                    }

    # TODO: add Methods.LLE config
    method_config_file = {Methods.PCA: {"n_components": None}}

    datasets = init_datasets()

    dimension_config_file = {dataset.name: list(range(1, len(dataset.data.columns) - 1)) for dataset in datasets}

    trainer = Trainer(classification_model_config_file, regression_model_config_file, datasets, method_config_file, dimension_config_file)
    trainer.train_all_combinations()

    results = trainer.process_results()
    print(results)
    print(results[(results.model == "Random Forest Regressor") & (results.dataset == "ames_housing")])
    results.to_csv("data/results.csv")


if __name__ == "__main__":
    main()
