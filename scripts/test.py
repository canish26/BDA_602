import random
import pandas as pd
import seaborn as sns
from sklearn import datasets

#defined as dictionaries within the class
class Test_Dataset:
    SEABORN_DATASETS = {
        "mpg": {
            "predictors": ["cylinders", "displacement", "horsepower", "weight", "acceleration", "origin"],
            "response": "mpg"
        },
        "tips": {
            "predictors": ["total_bill", "sex", "smoker", "day", "time", "size"],
            "response": "tip"
        },
        "titanic": {
            "predictors": [
                "pclass", "sex", "age", "sibsp", "embarked", "parch", "fare", "who",
                "adult_male", "deck", "embark_town", "alone", "class"
            ],
            "response": "survived"
        }
    }

    SKLEARN_DATASETS = {
        "diabetes": {
            "data": datasets.load_diabetes(),
            "predictors": datasets.load_diabetes().feature_names,
            "response": "target"
        },
        "breast_cancer": {
            "data": datasets.load_breast_cancer(),
            "predictors": datasets.load_breast_cancer().feature_names,
            "response": "target"
        }
    }

    def __init__(self):
        self.seaborn_data_sets = list(Test_Dataset.SEABORN_DATASETS.keys())
        self.sklearn_data_sets = list(Test_Dataset.SKLEARN_DATASETS.keys())
        self.all_data_sets = self.seaborn_data_sets + self.sklearn_data_sets

    def get_all_available_datasets(self):
        return self.all_data_sets

    def get_test_dataset(self, data_set_name=None):
        if data_set_name is None:
            data_set_name = random.choice(self.all_data_sets)
        elif data_set_name not in self.all_data_sets:
            raise Exception(f"Data set choice not valid: {data_set_name}")

        if data_set_name in Test_Dataset.SEABORN_DATASETS:
            data_set = sns.load_dataset(name=data_set_name)
            data_set = data_set.dropna().reset_index()
            predictors = Test_Dataset.SEABORN_DATASETS[data_set_name]["predictors"]
            response = Test_Dataset.SEABORN_DATASETS[data_set_name]["response"]
        elif data_set_name in Test_Dataset.SKLEARN_DATASETS:
            data = Test_Dataset.SKLEARN_DATASETS[data_set_name]["data"]
            data_set = pd.DataFrame(data.data, columns=data.feature_names)
            data_set["target"] = data.target
            predictors = Test_Dataset.SKLEARN_DATASETS[data_set_name]["predictors"]
            response = Test_Dataset.SKLEARN_DATASETS[data_set_name]["response"]

        # Change category dtype to string
        for predictor in predictors:
            if data_set[predictor].dtype in ["category"]:
                data_set[predictor] = data_set[predictor].astype(str)

        print(f"Data set selected: {data_set_name}")
        data_set.reset_index(drop=True, inplace=True)
        return data_set, predictors, response

if __name__ == "__main__":
    test_datasets = Test_Dataset()
    df_list = [
        [df, predictors, response]
        for df, predictors, response in [
            test_datasets.get_test_dataset(data_set_name=test)
            for test in test_datasets.get_all_available_datasets()
        ]
    ]
