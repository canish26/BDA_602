import random

import numpy as np
import pandas as pd
import plotly.express as px

# import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from sklearn import datasets


# used the code from class with some changes
class Test_Dataset:
    def __init__(self):
        self.seaborn_data_sets = ["mpg", "tips", "titanic"]
        self.sklearn_data_sets = ["diabetes", "breast_cancer"]
        self.all_data_sets = self.seaborn_data_sets + self.sklearn_data_sets

    TITANIC_PREDICTORS = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "embarked",
        "parch",
        "fare",
        "who",
        "adult_male",
        "deck",
        "embark_town",
        "alone",
        "class",
    ]

    def get_all_available_datasets(self):
        return self.all_data_sets

    def get_test_dataset(self, data_set_name=None):
        if data_set_name is None:
            data_set_name = random.choice(self.all_data_sets)
        else:
            if data_set_name not in self.all_data_sets:
                raise Exception(f"Data set choice not valid: {data_set_name}")

        if data_set_name in self.seaborn_data_sets:
            data_set = sns.load_dataset(name=data_set_name)
            data_set = data_set.dropna().reset_index()
            if data_set_name == "mpg":
                predictors = [
                    "cylinders",
                    "displacement",
                    "horsepower",
                    "weight",
                    "acceleration",
                    "origin",
                ]
                response = "mpg"
            elif data_set_name == "tips":
                predictors = [
                    "total_bill",
                    "sex",
                    "smoker",
                    "day",
                    "time",
                    "size",
                ]
                response = "tip"
            elif data_set_name in ["titanic", "titanic_2"]:
                data_set["alone"] = data_set["alone"].astype(str)
                data_set["class"] = data_set["class"].astype(str)
                data_set["deck"] = data_set["deck"].astype(str)
                data_set["pclass"] = data_set["pclass"].astype(str)
                predictors = self.TITANIC_PREDICTORS
                if data_set_name == "titanic":
                    response = "survived"
                elif data_set_name == "titanic_2":
                    response = "alive"
        elif data_set_name in self.sklearn_data_sets:
            if data_set_name == "boston":
                data = datasets.load_boston()
            elif data_set_name == "diabetes":
                data = datasets.load_diabetes()
            elif data_set_name == "breast_cancer":
                data = datasets.load_breast_cancer()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)
            data_set["target"] = data.target
            predictors = data.feature_names
            response = "target"

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


def cat_resp_cat_pred(data_set, pred_col, resp_col):
    pivoted_data = data_set.pivot_table(
        index=resp_col, columns=pred_col, aggfunc="size"
    )
    fig = px.imshow(
        pivoted_data.values,
        labels=dict(x=pred_col, y=resp_col),
        x=pivoted_data.columns,
        y=pivoted_data.index,
        color_continuous_scale="Blues",
    )
    # Creating figure
    fig.update_layout(title="Heatmap")
    # Show the plot
    fig.show()


def cat_resp_cont_pred(data_set, pred_col, resp_col):
    hist_data = [
        data_set[data_set[resp_col] == i][pred_col] for i in data_set[resp_col].unique()
    ]

    group_labels = (
        data_set[resp_col]
        .value_counts()
        .to_frame()
        .reset_index()["index"]
        .astype("string")
    )

    # created using the px.histogram() function from Plotly Express
    # Create distribution plot with custom bin_size
    fig_1 = px.histogram(
        data_set, x=pred_col, color=resp_col, histnorm="probability density"
    )
    fig_1.update_layout(
        title="Categorical " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=pred_col,
        yaxis_title="Probability Density",
    )
    fig_1.show()

    # added go.Violin() trace for each unique value in the predictor column
    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=[curr_group] * len(curr_hist),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title="Categorical " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=resp_col,
        yaxis_title=pred_col,
    )
    fig_2.show()


def cont_resp_cat_pred(data_set, pred_col, resp_col):
    # Group data together
    hist_data = [
        data_set[data_set[pred_col] == i][resp_col] for i in data_set[pred_col].unique()
    ]

    group_labels = data_set[pred_col].unique()

    # created using the px.histogram() function from Plotly Express
    # Create distribution plot with custom bin_size
    fig_1 = px.histogram(
        data_set, x=resp_col, color=pred_col, histnorm="probability density"
    )
    fig_1.update_layout(
        title="Continuous " + resp_col + " vs " + " Categorical " + pred_col,
        xaxis_title=resp_col,
        yaxis_title="Probability Density",
    )
    fig_1.show()

    # added go.Violin() trace for each unique value in the predictor column
    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=[curr_group] * len(curr_hist),
                y=curr_hist,
                name=str(curr_group),
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title="Continuous " + resp_col + " vs " + " Categorical " + pred_col,
        xaxis_title=pred_col,
        yaxis_title=resp_col,
    )
    fig_2.show()


# used the px.scatter() function from Plotly Express
def cont_resp_cont_pred(data_set, pred_col, resp_col):
    fig = px.scatter(data_set, x=pred_col, y=resp_col, trendline="ols")
    fig.update_layout(
        title="Continuous " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=pred_col,
        yaxis_title=resp_col,
    )
    fig.show()


def ploting_graphs(df, pred_col, pred_type, resp_col, resp_type):
    predictors = {"Categorical": cat_resp_cat_pred, "Continuous": cat_resp_cont_pred}
    responders = {
        "Boolean": predictors[pred_type],
        "Continuous": {
            "Categorical": cont_resp_cat_pred,
            "Continuous": cont_resp_cont_pred,
        }[pred_type],
    }
    responders[resp_type](df, pred_col, resp_col)


df_list[2][1]

len(df_list[4][1])
data_set = df_list[4][0]
pred = df_list[4][1]
resp = df_list[4][2]

def get_response_type(data_set, resp):
    unique_values = data_set[resp].unique()
    if np.issubdtype(unique_values.dtype, np.number) and len(unique_values) > 2:
        return "Continuous"
    elif len(unique_values) == 2:
        return "Boolean"
    else:
        raise ValueError("Response variable has too many categories")

def get_predictor_type(data_set, col):
    if np.issubdtype(data_set[col].dtype, np.number):
        return "Continuous"
    else:
        return "Categorical"

for col in pred:
    pred_type = get_predictor_type(data_set, col)
    resp_type = get_response_type(data_set, resp)
    ploting_graphs(
        data_set,
        pred_col=col,
        pred_type=pred_type,
        resp_col=resp,
        resp_type=resp_type,
    )

df_list[4][0].iloc[:, 18:]
