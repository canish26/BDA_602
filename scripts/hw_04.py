import random
import os
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_regress_exog
import plotly.graph_objects as go
import seaborn as sns
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

# used the code from class with some changes
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


# if __name__ == "__main__":
#    df_list = []
#    test_datasets = Test_Dataset()
#    for test in test_datasets.get_all_available_datasets():
#        df, predictors, response = test_datasets.get_test_dataset(data_set_name=test)
#        df_list.append([df, predictors, response])
def main():
    test_datasets = Test_Dataset()
    df_list = [
        [df, predictors, response]
        for df, predictors, response in [
            test_datasets.get_test_dataset(data_set_name=test)
            # Change the dataset name above to get the results of dataset you wish from test datasets
            for test in test_datasets.get_all_available_datasets()
        ]
    ]

    df_list[2][1]

    len(df_list[4][1])
    data_set = df_list[4][0]
    pred = df_list[4][1]
    resp = df_list[4][2]

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

@staticmethod
def creating_plot_dir():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = f"{this_dir}/Plots"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def combine_html(one, two, three, optional=None):
    # Reading data from file1
    with open(one) as fp:
        data = fp.read()

    # Reading data from file2
    with open(two) as fp:
        data2 = fp.read()

    # Merging 2 files
    # To add the data of file2
    # from next line
    data += "\n"
    data += "<hr>"
    data += data2

    if optional is not None:
        with open(optional) as fp:
            data3 = fp.read()
        data += "<hr>"
        data += data3

    with open(three, "w") as fp:
        fp.write(data)

def cat_resp_cat_pred(data_set, pred_col, resp_col, weight_col=None):
    X = data_set[pred_col]
    y = data_set[resp_col]
    if weight_col is not None:
        w = data_set[weight_col]
    else:
        w = None

    # fit logistic regression model
    model = sm.Logit(y, sm.add_constant(X), w)
    results = model.fit(disp=False)

    # extract p-value and t-score for predictor variable
    p_value = results.pvalues[pred_col]
    t_score = results.tvalues[pred_col]

    print(f"P-value for {pred_col}: {p_value}")
    print(f"T-score for {pred_col}: {t_score}")

    # create grouped bar chart of predictor vs response variable
    grouped_data = data_set.groupby([pred_col, resp_col]).size().reset_index(name='count')
    fig = px.bar(grouped_data, x=pred_col, y='count', color=resp_col)
    fig.show()

    # calculate difference with mean of response
    mean_y = np.average(y, weights=w)
    diff_y = y - mean_y

    # plot weighted and unweighted histograms of differences
    fig_weighted = px.histogram(
        data_set,
        x=diff_y,
        color=pred_col,
        histnorm="probability density",
        marginal="rug",  # add rug plot to show distribution
        weights=w,  # add weight_col to weight the data
    )
    fig_weighted.update_layout(
        title="Difference with mean of " + resp_col + " (Weighted)",
        xaxis_title="Difference with mean of " + resp_col,
        yaxis_title="Probability Density",
    )
    fig_weighted.show()

    fig_unweighted = px.histogram(
        data_set,
        x=diff_y,
        color=pred_col,
        histnorm="probability density",
        marginal="rug",  # add rug plot to show distribution
    )
    fig_unweighted.update_layout(
        title="Difference with mean of " + resp_col + " (Unweighted)",
        xaxis_title="Difference with mean of " + resp_col,
        yaxis_title="Probability Density",
    )
    fig_unweighted.show()

   # fig.write_html(
    #    file="{}/Diff Plot {} and {}.html".format(write_dir, pred_col, resp_col),
     #   include_plotlyjs="cdn",
    #)

def cat_resp_cont_pred(data_set, pred_col, resp_col):
    # Fit linear regression model
    X = data_set[[pred_col]]  # predictor variable
    y = data_set[resp_col]  # response variable
    X = sm.add_constant(X)
    model_1 = sm.Logit(y, X).fit()  # Fit logistic regression model
    model = sm.OLS(y, sm.add_constant(X)).fit()

    # Print model summary output
    print(model_1.summary())
    # Print model summary output
    print(model.summary())

    # Extract p-value and t-score for the predictor variable
    p_value = model.pvalues[pred_col]
    t_score = model.tvalues[pred_col]

    # Extract p-value and t-score for the predictor variable
    p_value = model_1.pvalues[pred_col]
    t_score = model_1.tvalues[pred_col]

    # Print p-value and t-score
    print("p-value: {:.2f}".format(p_value))
    print("t-score: {:.2f}".format(t_score))

    # Create scatter plot with regression line
    fig = px.scatter(data_set, x=pred_col, y=resp_col, trendline="ols")
    fig.update_layout(
        title="Continuous " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=pred_col,
        yaxis_title=resp_col,
    )
    fig.show()

    # Create scatter plot with logistic regression curve
    x_range = np.linspace(X[pred_col].min(), X[pred_col].max(), 100)
    y_range = model_1.predict(sm.add_constant(x_range))
    fig = px.scatter(data_set, x=pred_col, y=resp_col, trendline="ols")
    fig.add_traces(px.line(x=x_range, y=y_range).data)
    fig.update_layout(
        title="Continuous " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=pred_col,
        yaxis_title=resp_col,
    )
    fig.show()

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
    # Calculate unweighted mean response
    unweighted_mean = data_set[resp_col].mean()

    # Calculate weighted mean response
    weighted_mean = data_set.groupby(pred_col)[resp_col].mean().mean()

    # Calculate difference in means
    diff_means = weighted_mean - unweighted_mean

    # Create distribution plot with custom bin_size
    fig_1 = px.histogram(
        data_set, x=pred_col, color=resp_col, histnorm="probability density"
    )
    fig_1.update_layout(
        title="Categorical " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=pred_col,
        yaxis_title="Probability Density",
    )
    # Add vertical lines for mean responses
    fig_1.add_shape(
        type="line",
        x0=unweighted_mean,
        y0=0,
        x1=unweighted_mean,
        y1=1,
        line=dict(color="black", dash="dash"),
    )
    fig_1.add_annotation(
        x=unweighted_mean,
        y=1.02,
        text="Unweighted Mean: {:.2f}".format(unweighted_mean),
        showarrow=False,
    )

    fig_1.update_layout(title_text="Histogram Plot with Mean Line")

    fig_1.show()

    # Create violin plot with weighted mean lines
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
    # Add vertical lines for mean responses
    fig_2.add_shape(
        type="line",
        x0=weighted_mean,
        y0=data_set[resp_col].min(),
        x1=weighted_mean,
        y1=data_set[resp_col].max(),
        line=dict(color="black", dash="dash"),
    )
    fig_2.add_annotation(
        x=weighted_mean,
        y=data_set[resp_col].max()
          + 0.1 * (data_set[resp_col].max() - data_set[resp_col].min()),
        text="Weighted Mean: {:.2f}".format(weighted_mean),
        showarrow=False,
    )

    fig_2.show()

    #fig.write_html(
     #   file=f"{write_dir}/Diff Plot {pred_col} and {resp_col}.html",
      #  include_plotlyjs="cdn",
    #)

    print("Difference in means: {:.2f}".format(diff_means))

#combine_html(
 #   optional=f"{write_dir}/Diff_plot_{pred_col}_and_{resp_col}.html",
  #  one=f"{write_dir}/Unweighted_Diff_Table_of_{pred_col}.html",
   #two=f"{write_dir}/Weighted_Diff_Table_of_{pred_col}.html",
#)

def cont_resp_cat_pred(data_set, pred_col, resp_col, write_dir):
    X = data_set[pred_col]
    y = data_set[resp_col]

    # add constant to predictor variables
    X = sm.add_constant(X)

    # fit linear regression model
    model = sm.OLS(y, X)
    results = model.fit()

    # fit logistic regression model
    model_1 = sm.Logit(y, X)
    results = model.fit()

    # extract p-value and t-score for predictor variable
    p_value = results.pvalues[pred_col]
    t_score = results.tvalues[pred_col]

    print(f"P-value for {pred_col}: {p_value}")
    print(f"T-score for {pred_col}: {t_score}")


    # Create scatter plot with regression line
    fig = px.scatter(data_set, x=pred_col, y=resp_col, trendline="ols")
    fig.update_layout(
        title="Continuous " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=pred_col,
        yaxis_title=resp_col,
    )
    fig.show()

    # create logistic regression plots
    fig = plot_regress_exog(results, pred_col)
    fig.tight_layout()
    fig.show()

    # Group data together
    hist_data = [
        data_set[data_set[pred_col] == i][resp_col] for i in data_set[pred_col].unique()
    ]

    group_labels = data_set[pred_col].unique()

    grouped_data = data_set.groupby(pred_col)
    unweighted_mean = grouped_data[resp_col].mean()
    weighted_mean = grouped_data.apply(
        lambda x: np.average(x[resp_col], weights=x["weight_col"])
    )

    mean_diff = weighted_mean - unweighted_mean

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

    fig_3 = px.bar(
        x=mean_diff.index,
        y=mean_diff.values,
        labels={"x": pred_col, "y": "Weighted mean - Unweighted mean"},
        title="Difference with mean of response",
    )
    fig_3.show()

    # unweighted histogram
    fig_4 = px.histogram(
        data_set,
        x=resp_col,
        color=pred_col,
        histnorm="probability density",
        marginal="rug",  # add rug plot to show distribution
    )
    fig_4.update_layout(
        title="Continuous "
        + resp_col
        + " vs "
        + " Categorical "
        + pred_col
        + " (Unweighted)",
        xaxis_title=resp_col,
        yaxis_title="Probability Density",
    )
    fig_4.show()

    # weighted histogram
    fig_5 = px.histogram(
        data_set,
        x=resp_col,
        color=pred_col,
        histnorm="probability density",
        marginal="rug",  # add rug plot to show distribution
        weights="weight_col",  # add weight_col to weight the data
    )
    fig_5.update_layout(
        title="Continuous "
        + resp_col
        + " vs "
        + " Categorical "
        + pred_col
        + " (Weighted)",
        xaxis_title=resp_col,
        yaxis_title="Probability Density",
    )
    fig_5.show()

    fig_6 = px.scatter(data_set, x=pred_col, y=resp_col)
    fig_6.update_layout(
        title="Continuous " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=pred_col,
        yaxis_title=resp_col,
    )
    fig_6.show()

   # fig.write_html(
    #    file="{}/Diff Plot {} and {}.html".format(write_dir, pred_col, resp_col),
     #   include_plotlyjs="cdn",
    #)


# used the px.scatter() function from Plotly Express
def cont_resp_cont_pred(data_set, pred_col, resp_col, weight_col=None):
    # Fit linear regression model
    X = data_set[[pred_col]]  # predictor variable
    y = data_set[resp_col]  # response variable
    if weight_col:
        w = data_set[weight_col]  # weight variable
        model = sm.WLS(y, sm.add_constant(X), weights=w).fit()
        mean_y = np.average(y, weights=w)
    else:
        model = sm.OLS(y, sm.add_constant(X)).fit()
        mean_y = np.mean(y)

        # Print model summary output
    print(model.summary())

    # Extract p-value and t-score for the predictor variable
    p_value = model.pvalues[pred_col]
    t_score = model.tvalues[pred_col]

    # Print p-value and t-score
    print("p-value: {:.2f}".format(p_value))
    print("t-score: {:.2f}".format(t_score))

    # Create scatter plot with regression line and mean line
    fig = px.scatter(data_set, x=pred_col, y=resp_col, trendline="ols")
    fig.add_shape(
        type="line",
        x0=data_set[pred_col].min(),
        y0=mean_y,
        x1=data_set[pred_col].max(),
        y1=mean_y,
        line=dict(color="red", width=2, dash="dash"),
    )

    fig.update_layout(
        title="Continuous " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=pred_col,
        yaxis_title=resp_col,
    )
    fig.show()

    # Create weighted scatter plot with regression line and mean line
    if weight_col:
        fig = px.scatter(
            data_set, x=pred_col, y=resp_col, weight=weight_col, trendline="ols"
        )
        fig.add_shape(
            type="line",
            x0=data_set[pred_col].min(),
            y0=mean_y,
            x1=data_set[pred_col].max(),
            y1=mean_y,
            line=dict(color="red", width=2, dash="dash"),
        )

        fig.update_layout(
            title="Weighted Continuous " + resp_col + " vs " + " Continuous " + pred_col,
            xaxis_title=pred_col,
            yaxis_title=resp_col,
        )
        fig.show()

        #fig.write_html(
         #   file="{}/Diff Plot {} and {}.html".format(write_dir, pred_col, resp_col),
          #  include_plotlyjs="cdn",
        #)

def bool_resp_cont_pred(data_set, pred_col, resp_col, write_dir):
    # Fit logistic regression model
    X = data_set[[pred_col]]  # predictor variable
    y = data_set[resp_col]  # response variable
    model = sm.Logit(y, sm.add_constant(X)).fit(disp=False)

    # Print model summary output
    print(model.summary())

    # Extract p-value and t-score for the predictor variable
    p_value = model.pvalues[pred_col]
    t_score = model.tvalues[pred_col]

    # Print p-value and t-score
    print("p-value: {:.2f}".format(p_value))
    print("t-score: {:.2f}".format(t_score))

    # Create scatter plot with logistic regression curve
    x_vals = np.linspace(data_set[pred_col].min(), data_set[pred_col].max(), 100)
    log_odds = model.params[0] + model.params[1] * x_vals
    odds = np.exp(log_odds)
    p = odds / (1 + odds)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data_set[pred_col],
            y=data_set[resp_col],
            mode="markers",
            name="Data",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=p,
            mode="lines",
            name="Logistic Regression Curve",
        )
    )

    fig.update_layout(
        title="Boolean " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=pred_col,
        yaxis_title=resp_col,
    )
    fig.show()

def cont_resp_bool_pred(data_set, pred_col, resp_col, write_dir):
    # Fit logistic regression model
    X = data_set[[pred_col]]  # predictor variable
    y = data_set[resp_col]  # response variable
    model = sm.Logit(y, sm.add_constant(X)).fit()

    # Print model summary output
    print(model.summary())

    # Extract p-value and z-score for the predictor variable
    p_value = model.pvalues[pred_col]
    z_score = model.tvalues[pred_col]

    # Print p-value and z-score
    print("p-value: {:.2f}".format(p_value))
    print("z-score: {:.2f}".format(z_score))

    # Create scatter plot with logistic regression line
    fig = px.scatter(data_set, x=pred_col, y=resp_col, trendline="ols")
    fig.update_layout(
        title="Boolean " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=pred_col,
        yaxis_title=resp_col,
    )
    fig.add_trace(
        go.Scatter(
            x=data_set[pred_col],
            y=model.predict(sm.add_constant(data_set[pred_col])),
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Logistic Regression Line",
        )
    )
    fig.show()

#generating a table with variable importance rankings for a random forest model with continuous predictors only
def cont_pred_rf_variable_importance(data_set, pred_cols, resp_col):
    # Fit random forest model
    X = data_set[pred_cols]
    y = data_set[resp_col]
    model = RandomForestRegressor()
    model.fit(X, y)

    # Get variable importance rankings
    importances = model.feature_importances_
    var_names = X.columns
    rankings = pd.DataFrame({'Variable': var_names, 'Importance': importances})
    rankings = rankings.sort_values('Importance', ascending=False)

    # Print rankings table
    print(rankings)

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

 #def make_clickable(val):
        """Make urls in dataframe clickable for html output"""

  #      if val is not None:
   #         if "," in val:
    #            x = val.split(",")
     #           return f'{x[0]} <a target="_blank" href="{x[1]}">link to plot</a>'
      #      else:
       #         return f'<a target="_blank" href="{val}">link to plot</a>'
        #else:
         #   return val

if __name__ == "__main__":
    main()

