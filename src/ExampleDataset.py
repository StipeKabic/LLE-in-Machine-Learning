import pandas as pd


class ExampleDataset:
    def __init__(self):
        self.name = "Example dataset"
        self.target_name = "target"
        self.data = pd.DataFrame({"feature1": list(range(100)),
                                  "feature2": list(range(100)),
                                  "feature3": list(range(100)),
                                  "target": 30 * [0] + 40*[1] + 30*[0]})
