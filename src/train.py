from sklearn.model_selection import train_test_split
from ExampleDataset import ExampleDataset
from enums import Models, Methods
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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
        pca.fit_transform(x)
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
    def __init__(self, model_config, datasets, method_config):
        """
        Initialize trainer class
        :param model_config: config for model hyperparameters
        :param datasets: list of datasets - dataset is a class with attributes: name(string), data(pandas dataframe), target_name(string, name of target variable)
        :param method_config: config with hyperparameters for dimensionality reduciton methods
        """
        self.models = {model: self.init_models(model)(**model_config[model]) for model in model_config}
        self.datasets = datasets
        self.method_config = method_config
        self.results = {}

    def init_models(self, model):
        if model == Models.Logistic:
            return LogisticRegression
        elif model == Models.RF:
            return RandomForestClassifier
        else:
            raise NotImplementedError

    def train(self, model_name, dataset):
        accuracies = {}
        model = self.models[model_name]
        for method in (Methods.PCA, Methods.Full):
            split = split_dataset(dataset, method, self.method_config)
            model.fit(split.x_train, split.y_train)
            accuracy = model.score(split.x_test, split.y_test)
            accuracies[method.value] = accuracy
        return accuracies

    def train_all_combinations(self):
        for model_name in self.models:
            for dataset in self.datasets:
                results = self.train(model_name, dataset)
                self.results[(model_name.value, dataset.name)] = results


def main():
    model_config_file = {Models.Logistic: {"penalty": 'none'},
                         Models.RF: {"n_estimators": 10,
                                     "max_depth": 5,
                                     "min_samples_split": 2,
                                     "min_samples_leaf": 1,
                                     "max_features": "sqrt"}}

    # TODO: add Methods.LLE config
    method_config_file = {Methods.PCA: {"n_components": 1}}

    dataset = ExampleDataset()
    datasets = [dataset]
    trainer = Trainer(model_config_file, datasets, method_config_file)
    print(trainer.results)
    trainer.train_all_combinations()
    print(trainer.results)


if __name__ == "__main__":
    main()
