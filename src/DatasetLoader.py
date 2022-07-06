import sklearn.datasets as sd
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DatasetLoader:
    def __init__(self):
        self.datasets = {}
        self.load_iris()
        self.load_california_housing()
        self.load_ames_housing()
        self.load_heart_attack()
        self.load_stroke()

        self.names = list(self.datasets.keys())

    def load_iris(self):
        # klasik iris dataset, ogledni primjer
        data = sd.load_iris(as_frame=True)
        df = data['data']
        df['target'] = data['target']
        self.datasets['iris'] = {'data': df, 'target_name': 'target', 'type': 'c'}

    def load_california_housing(self):
        # https://gist.github.com/machinelearning-blog/76b50b18c7db3408646cc8d18c50c20b
        data = sd.fetch_california_housing(as_frame=True)
        df = data['data']
        df['house_value'] = data['target']
        self.datasets['california_housing'] = {'data': df, 'target_name': 'house_value', 'type': 'r'}

    def load_ames_housing(self):
        data = sd.fetch_openml(name="house_prices", as_frame=True)
        df = data['data']
        df['sale_price'] = data['target']
        categorical = df.select_dtypes(include=['object']).columns
        # df = pd.get_dummies(df, columns=categorical)
        # df.dropna(inplace=True)
        label_encoder = LabelEncoder()
        for categorical_variable in categorical:
            df[categorical_variable] = label_encoder.fit_transform(df[categorical_variable])
        df.dropna(inplace=True)
        self.datasets['ames_housing'] = {'data': df, 'target_name': 'sale_price', 'type': 'r'}

    def load_heart_attack(self):
        # https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
        df = pd.read_csv('../data/heart.csv')
        df.rename(columns={'output': 'target'}, inplace=True)
        self.datasets['heart_attack'] = {'data': df, 'target_name': 'target', 'type': 'c'}

    def load_stroke(self):
        # https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
        df = pd.read_csv('../data/healthcare-dataset-stroke-data.csv')
        # df.rename(columns={'stroke' : 'target'}, inplace=True)
        df.dropna(inplace=True)
        df = pd.get_dummies(df, columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"])
        df.drop(columns=["id"], inplace=True)
        self.datasets['stroke'] = {'data': df, 'target_name': 'stroke', 'type': 'c'}
