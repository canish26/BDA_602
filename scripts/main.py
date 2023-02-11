# from urllib.request import urlretrieve
# import chart_studio.plotly as py

# import numpy as np
import pandas as pd

# from plotly.offline import init_notebook_mode, iplot
# from sklearn.datasets import load_iris
import plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go

# Data Split Libraries
# import sklearn
import spark as spark

# import plotly.graph_objects as go
from pandas import DataFrame
from plotly.offline import init_notebook_mode, iplot

# Apache Spark Pipelin Library
from pyspark.ml import Pipeline

# Apache Spark ML CLassifier Libraries
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
)

# Apache Spark Evaluation Library
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Apache Spark Features libraries
from pyspark.ml.feature import StandardScaler, StringIndexer

# Apache Spark `DenseVector`
from pyspark.ml.linalg import DenseVector

# from pyspark.sql import SparkSession
from sklearn.decomposition import PCA

# Tabulating Data
from tabulate import tabulate

# Apache Spark Libraries
# import pyspark


# from sklearn.model_selection import train_test_split


init_notebook_mode(connected=True)
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

iris = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(iris, sep=",")
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "Species"]
df.columns = attributes

iris_obj = iris()
# Dataset preview
iris_obj.data  # Names of the columns
iris_obj.feature_names  # Target variable
iris_obj.target  # Target names
iris_obj.target_names  # name of target variable

set(df["Species"])
df["Species"].value_counts()
df_setosa = df[df["Species"] == "Iris-setosa"]
df_virginica = df[df["Species"] == "Iris-virginica"]
df_versicolor = df[df["Species"] == "Iris-versicolor"]

setosa = go.Scatter(
    x=df["sepal_length"][df.Species == "Iris-setosa"],
    y=df["sepal_width"][df.Species == "Iris-setosa"],
    mode="markers",
    name="setosa",
)
versicolor = go.Scatter(
    x=df["sepal_length"][df.Species == "Iris-versicolor"],
    y=df["sepal_width"][df.Species == "Iris-versicolor"],
    mode="markers",
    name="versicolor",
)
virginica = go.Scatter(
    x=df["sepal_length"][df.Species == "Iris-virginica"],
    y=df["sepal_width"][df.Species == "Iris-virginica"],
    mode="markers",
    name="virginica",
)
data = [setosa, versicolor, virginica]
fig = dict(data=data)
py.iplot(fig, filename="styled-scatter")

# boxplot_2
trace1 = go.Scatter(
    y=df_setosa["sepal_length"],
    mode="markers",
    marker=dict(
        size=16,
        color=300,  # set color equal to a variable
        colorscale="Viridis",
        showscale=True,
    ),
)

trace2 = go.Scatter(
    y=df_setosa["sepal_width"],
    mode="markers",
    marker=dict(
        size=16,
        color=200,  # set color equal to a variable
        colorscale="Viridis",
        showscale=True,
    ),
)


data = [trace1, trace2]

py.iplot(data, filename="scatter-plot-with-colorscale")

# Petal Length and Width with each target variable
setosa = go.Scatter(
    x=df["petal_length"][df.Species == "Iris-setosa"],
    y=df["petal_width"][df.Species == "Iris-setosa"],
    mode="markers",
    name="setosa",
)
versicolor = go.Scatter(
    x=df["petal_length"][df.Species == "Iris-versicolor"],
    y=df["petal_width"][df.Species == "Iris-versicolor"],
    mode="markers",
    name="versicolor",
)
virginica = go.Scatter(
    x=df["petal_length"][df.Species == "Iris-virginica"],
    y=df["petal_width"][df.Species == "Iris-virginica"],
    mode="markers",
    name="virginica",
)
data = [setosa, versicolor, virginica]
fig = dict(data=data)
py.iplot(fig, filename="styled-scatter")

# Boxplot plotting
trace0 = go.Box(
    y=df["petal_width"][df["Species"] == "Iris-setosa"], boxmean=True, name="setosa"
)

trace1 = go.Box(
    y=df["petal_width"][df["Species"] == "Iris-versicolor"],
    boxmean=True,
    name="versicolor",
)

trace2 = go.Box(
    y=df["petal_width"][df["Species"] == "Iris-virginica"],
    boxmean=True,
    name="virginica",
)

data = [trace0, trace1, trace2]
py.iplot(data)

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
# Scatterplot Matrix
data_ff = data.loc[
    :, ["sepal length (cm)", "sepal width (cm)", "petal length(cm)", "petal width (cm)"]
]
data_ff["index"] = ff.np.arange(1, len(data_ff) + 1)

fig_ff = ff.create_scatterplotmatrix(
    data_ff,
    diag="box",
    index="index",
    colormap="Blues",
    colormap_type="cat",
    height=800,
    width=800,
)
iplot(fig_ff)

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

# model building
# String Indexing the Species column
SIndexer = StringIndexer(inputCol="species", outputCol="species_indx")
data = SIndexer.fit(data).transform(data)
# Inspect the dataset
data.show(5)
input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
# Creating a new Indexed Dataframe
df_indx = spark.createDataFrame(input_data, ["label", "features"])
# Initialize Standard Scaler
stdScaler = StandardScaler(inputCol="features", outputCol="features_scaled")

# Fit the Standard Scaler to the indexed Dataframe
scaler = stdScaler.fit(df_indx)

# Transform the dataframe
df_scaled = scaler.transform(df_indx)
df_scaled = df_scaled.drop("features")
train_data, test_data = df_scaled.randomSplit([0.9, 0.1], seed=12345)
# Inspect Training Data
train_data.show(5)

model = ["Decision Tree", "Random Forest", "Naive Bayes"]
model_results = []

# Pipeline Creation Data preprocessing

pipeline_lr = Pipeline(
    [
        ("scaler1", StandardScaler()),
        ("pca1", PCA(n_components=3)),
        ("lr_classifier", LogisticRegression(random_state=0)),
    ]
)
pipeline_dt = Pipeline(
    [
        ("scaler2", StandardScaler()),
        ("pca2", PCA(n_components=3)),
        ("lr_classifier", DecisionTreeClassifier()),
    ]
)
pipeline_rf = Pipeline(
    [
        ("scaler3", StandardScaler()),
        ("pca3", PCA(n_components=3)),
        ("lr_classifier", RandomForestClassifier()),
    ]
)
pipeline_rf = Pipeline(
    [
        ("scaler3", StandardScaler()),
        ("pca3", PCA(n_components=3)),
        ("lr_classifier", RandomForestClassifier()),
    ]
)
# Let's make the list of pipelines
pipelines = [pipeline_lr, pipeline_dt, pipeline_rf]

best_accuracy = 0.0
best_classifier = 0
best_pipeline = ""

# -- Decision Tree Classifier --

dtc = DecisionTreeClassifier(
    labelCol="label", featuresCol="features_scaled"
)  # instantiate the model
dtc_model = dtc.fit(train_data)  # train the model
dtc_pred = dtc_model.transform(test_data)  # model predictions

# Evaluate the Model
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
dtc_acc = evaluator.evaluate(dtc_pred)
# print("Decision Tree Classifier Accuracy =", '{:.2%}'.format(dtc_acc))
model_results.extend([[model[0], "{:.2%}".format(dtc_acc)]])  # appending to list

# -- Random Forest Classifier --

rfc = RandomForestClassifier(
    labelCol="label", featuresCol="features_scaled", numTrees=10
)  # instantiate the model
rfc_model = rfc.fit(train_data)  # train the model
rfc_pred = rfc_model.transform(test_data)  # model predictions

# Evaluate the Model
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
rfc_acc = evaluator.evaluate(rfc_pred)
# print("Random Forest Classifier Accuracy =", '{:.2%}'.format(rfc_acc))
model_results.extend([[model[1], "{:.2%}".format(rfc_acc)]])  # appending to list

# -- Naive Bayes Classifier --

nbc = NaiveBayes(
    smoothing=1.0,
    modelType="multinomial",
    labelCol="label",
    featuresCol="features_scaled",
)  # instantiate the model
nbc_model = nbc.fit(train_data)  # train the model
nbc_pred = nbc_model.transform(test_data)  # model predictions

# Evaluate the Model
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
nbc_acc = evaluator.evaluate(nbc_pred)
# print("Naive Bayes Accuracy =", '{:.2%}'.format(nbc_acc))
model_results.extend([[model[2], "{:.2%}".format(nbc_acc)]])

print(tabulate(model_results, headers=["Classifier Models", "Accuracy"]))
