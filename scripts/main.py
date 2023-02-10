# from urllib.request import urlretrieve

# import numpy as np
import pandas as pd

# import plotly.graph_objects as go
from pandas import DataFrame

# from plotly.offline import init_notebook_mode, iplot
from sklearn.datasets import load_iris

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

iris = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(iris, sep=",")
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df.columns = attributes

iris_obj = load_iris()
# Dataset preview
# iris_obj.data  # Names of the columns
# iris_obj.feature_names  # Target variable
# iris_obj.target  # Target names
# iris_obj.target_names  # name of target variable


iris = DataFrame(
    iris_obj.data,
    columns=iris_obj.feature_names,
    index=pd.Index([i for i in range(iris_obj.data.shape[0])]),
).join(
    DataFrame(
        iris_obj.target,
        columns=pd.Index(["species"]),
        index=pd.Index([i for i in range(iris_obj.target.shape[0])]),
    )
)
# iris  # prints iris data
iris.species.replace({0: "setosa", 1: "versicolor", 2: "virginica"}, inplace=True)
# iris  # prints labeled data

setosa = iris[iris["species"] == "Iris-setosa"]
versicolor = iris[iris["species"] == "Iris-versicolor"]
virginica = iris[iris["species"] == "Iris-virginica"]


iris_grps = iris.groupby("species")
for name, data in iris_grps:
    print(name)
    print(data.iloc[:, 0:4])
    print(iris.count())
    print(iris.mean())
    print(iris.median())
    print(iris.max())
    print(iris.std())
    print(iris.quantile(0.95))
    print(iris.quantile(0.75))
    print(iris.quantile(0.25))
    print(iris.quantile(0.75) - iris.quantile(0.25))
    print(iris_grps.mean())
    print(iris_grps.quantile(0.75) - iris_grps.quantile(0.25))


def range_stat(s):
    return s.max() - s.min()


iris.iloc[:, 0:4].apply(range_stat)
print(iris_grps.aggregate(range_stat))
