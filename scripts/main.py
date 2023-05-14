import itertools
import os
import sys

import numpy as np
import pandas as pd
import path as p
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.api as sm
from correlation_plots import Correlation_Preds as pc

# from dataset_loader import Test_Dataset
from pandas.core.dtypes.common import is_numeric_dtype
from plots import Plots as mp
from pred_resp_graphs import PlotGraph as pg
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from utilities import maria_db, model_results

path = p.GLOBAL_PATH
os.makedirs(path, exist_ok=True)

path_2d_morp = p.GLOBAL_PATH_2D_MORP
os.makedirs(os.path.join(path_2d_morp), exist_ok=True)


def rf_rank_cont_resp(df, cont_var_pred_list, cat_var_pred_list, resp):
    x_cont = df[cont_var_pred_list]
    x_cat = df[cat_var_pred_list]
    y = df[resp]

    # Encode categorical variables
    x_cat = x_cat.apply(LabelEncoder().fit_transform)

    # Concatenate continuous and categorical features
    df = pd.concat([x_cont, x_cat], axis=1)

    # Train random forest model
    rf = RandomForestRegressor(n_estimators=150)
    rf.fit(df, y)

    # Rank feature importance and create a DataFrame
    feature_importance = pd.DataFrame(
        {
            "Column_name": df.columns,
            "Column_type": np.where(df.columns.isin(x_cont.columns), "Cont", "Cat"),
            "fet_imp_coeff": rf.feature_importances_,
        }
    )

    return feature_importance.sort_values("fet_imp_coeff", ascending=False).reset_index(
        drop=True
    )


def rf_rank_cat_resp(df, cont_var_pred_list, cat_var_pred_list, resp):
    x_cont = df[cont_var_pred_list]
    x_cat = df[cat_var_pred_list]
    y = df[resp]

    encoder = LabelEncoder()
    x_cat_encoded = x_cat.apply(encoder.fit_transform)

    df = pd.concat([x_cont, x_cat_encoded], axis=1)

    # Train random forest model
    rf = RandomForestClassifier(n_estimators=150)
    rf.fit(df, y)

    # Rank feature importance and create a DataFrame
    rank_list = [
        {
            "Column_name": i,
            "Column_type": "Cont" if i in x_cont.columns else "Cat",
            "fet_imp_coeff": j,
        }
        for i, j in zip(df, rf.feature_importances_)
    ]

    rank_list_df = pd.DataFrame(rank_list)
    return rank_list_df.sort_values("fet_imp_coeff", ascending=False).reset_index(
        drop=True
    )


def lin_reg_plots(y, x, fet_nm):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()

    # Get the stats
    t_value = round(model.tvalues[1], 6)
    p_value = "{:.6e}".format(model.pvalues[1])
    p_value = np.float64(p_value)

    return {
        "Column_name": fet_nm,
        "Column_type": "Cont",
        "P_value": p_value,
        "T_value": t_value,
    }


def log_reg_plots(y, X, fet_nm):
    # Add a column of ones to the feature matrix X to represent the intercept
    X = sm.add_constant(X)

    # Fit a logistic regression model
    log_reg_model = sm.Logit(y, X).fit()

    # Get the stats
    t_value = round(log_reg_model.tvalues[1], 6)
    p_value = "{:.6e}".format(log_reg_model.pvalues[1])
    p_value = np.float64(p_value)

    return {
        "Column_name": fet_nm,
        "Column_type": "Cont",
        "P_value": p_value,
        "T_value": t_value,
    }


def cat_cat_2d_morp(df_ip, x1, x2, y):
    df = df_ip.groupby([x1, x2])[y].agg(["mean", "size"]).reset_index()
    df.columns = [x1, x2, "mean", "size"]

    df["unweigh_morp"] = (df[y]["mean"].mean() - df["mean"]) ** 2
    df["weigh_morp"] = (df["size"] / df["size"].sum()) * df["unweigh_morp"]
    mean_size = df.apply(lambda row: f"{row['mean']:.6f} pop:{row['size']}", axis=1)

    fig_heatmap = px.imshow(
        df.pivot(index=x1, columns=x2, values="mean"),
        x=df[x2].unique(),
        y=df[x1].unique(),
        color_continuous_scale="YlGnBu",
        labels={
            "x": x2.replace("_bin", ""),
            "y": x1.replace("_bin", ""),
            "color": "Corr",
        },
        title=f"{x2.replace('_bin', '')} vs {x1.replace('_bin', '')}",
    )

    fig_heatmap.update_traces(
        text=mean_size.values,
        texttemplate="%{text}",
        hovertemplate="%{x}<br>%{y}<br>Corr: %{z:.6f}<br>Population: %{text}",
    )

    fig_heatmap.write_html(
        file=f"{path_2d_morp}/cat_{x1}_cat_{x2}_2D_morp.html", include_plotlyjs="cdn"
    )

    return {
        "Weigh_morp": df["weigh_morp"].sum(),
        "Unweigh_morp": df["unweigh_morp"].sum()
                        / (df_ip[x1].nunique() * df_ip[x2].nunique()),
        "Plot_link": f"{path_2d_morp}/cat_{x1}_cat_{x2}_2D_morp.html",
    }


def cat_cont_2d_morp(df_ip, x1, x2, y):
    df = df_ip.copy()
    x2_bin = x2 + "_bin"
    df[x2_bin] = pd.cut(df[x2], bins=10)  # assuming 10 bins

    df_grouped = df.groupby([x1, x2_bin])[y].agg(["mean", "size"]).reset_index()
    df_grouped.rename(columns={"mean": y + "_mean", "size": y + "_size"}, inplace=True)

    unweighted_morp = (df[y].mean() - df_grouped[y + "_mean"]) ** 2
    weighted_morp = (
                            df_grouped[y + "_size"] / df_grouped[y + "_size"].sum()
                    ) * unweighted_morp
    mean_size = df_grouped.apply(
        lambda row: f"{row[y + '_mean']:.6f} pop:{row[y + '_size']}", axis=1
    )

    fig = px.imshow(
        df_grouped.pivot(x1, x2_bin, y + "_mean"),
        color_continuous_scale="YlGnBu",
        labels=dict(x=x1.replace("_bin", ""), y=x2.replace("_bin", ""), color="Corr"),
        title=f"{x2.replace('_bin', '')} vs {x1.replace('_bin', '')}",
    )
    fig.update_traces(text=mean_size.values, texttemplate="%{text}")
    fig.write_html(
        file=f"{path_2d_morp}/cat_{x1}_cont_{x2_bin}_2D_morp.html",
        include_plotlyjs="cdn",
    )

    return {
        "Weigh_morp": weighted_morp.sum(),
        "Unweigh_morp": unweighted_morp.sum() / len(df_grouped),
        "Plot_link": f"{path_2d_morp}/cat_{x1}_cont_{x2_bin}_2D_morp.html",
    }


def cont_cont_2d_morp(df_ip, x1, x2, y):
    # Create new columns with bin names
    df_ip[x1 + "_bin"] = pd.cut(df_ip[x1], bins=10)
    df_ip[x2 + "_bin"] = pd.cut(df_ip[x2], bins=10)

    # Group by bin columns and calculate mean and size for y
    groupby_cols = [x1 + "_bin", x2 + "_bin"]
    df = df_ip.groupby(groupby_cols).agg({y: ["mean", "size"]}).reset_index()
    df.columns = ["".join(col) for col in df.columns.to_flat_index()]

    # Calculate unweighted and weighted morp
    y_mean = df[y + "mean"]
    df["unweigh_morp"] = (df_ip[y].mean() - y_mean) ** 2
    df["weigh_morp"] = (df[y + "size"] / df[y + "size"].sum()) * df["unweigh_morp"]

    # Create mean_size column
    df["mean_size"] = df.apply(
        lambda row: f"{row[y + 'mean']:.6f} pop:{row[y + 'size']}", axis=1
    )

    # Create heatmap data and layout
    fig_heatmap = px.imshow(
        df,
        x=x1 + "_bin",
        y=x2 + "_bin",
        z=y + "mean",
        color_continuous_scale="YlGnBu",
        title=f"{x2.replace('_bin', '')} vs {x1.replace('_bin', '')}",
        labels=dict(x=x1, y=x2, z="Correlation"),
        text=df["mean_size"],
        hovertemplate="(%{x}, %{y})<br>Correlation: %{z:.2f}<br>%{text}",
        width=800,
        height=800,
    )

    # Create and save heatmap figure
    file_name = f"{path_2d_morp}/cont_{x1}_cont_{x2}_2D_morp.html"
    fig_heatmap.write_html(file=file_name, include_plotlyjs="cdn")

    # Return morp values and plot link
    return {
        "Weigh_morp": df["weigh_morp"].sum(),
        "Unweigh_morp": df["unweigh_morp"].sum() / len(df),
        "Plot_link": file_name,
    }


def pred_typ(data_set, pred_list):
    return {
        i: "Cat"
        if type(data_set[i][0]) == str
           or data_set[i].nunique() == 2
           and not data_set[i].dtype.kind in "iufc"
        else "Cont"
        for i in pred_list
    }


def url_click(url):
    return (
        f'<a target="_blank" href="{url.split(",")[1] if "," in url else url}">plots link</a>'
        if url
        else ""
    )


def preprocess_data(df):
    # Sort by game date and game id
    df.sort_values(by=["game_date", "game_id"], inplace=True, ignore_index=True)

    # Set game id as index
    df.set_index("game_id", inplace=True)

    # Split into train and test sets
    train_size = int(len(df) * 0.6)
    train = df.iloc[:train_size, 1:]
    test = df.iloc[train_size:, 1:]

    # Get column names
    cols = train.columns

    # Replace nulls with medians of training set
    medians = train.median(axis=0, skipna=True)
    for col_idx, col in enumerate(cols):
        train[col] = train[col].fillna(medians[col_idx])
        test[col] = test[col].fillna(medians[col_idx])

    # Combine train and test sets
    df = pd.concat([train, test])

    return df


def select_features(df):
    # Select relevant features
    features = [
        "SP_BFP_DIFF_ROLL",
        "SP_SO9_DIFF_ROLL",
        "TB_OPS_DIFF_HIST",
        "TB_BABIP_DIFF_ROLL",
    ]
    X = df[features]
    y = df.iloc[:, -1]

    return X, y


def fit_stats_model(x, y):
    pred = sm.add_constant(x)
    logr_model = sm.Logit(y, pred)
    logr_model_fitted = logr_model.fit()
    print(logr_model_fitted.summary())
    return logr_model_fitted


def split_data(df, features, train_size):
    x = df[features]
    y = df["HTWins"]
    x_train = x.iloc[:train_size, :].values
    x_test = x.iloc[train_size:, :].values
    y_train = df.iloc[:train_size, -1].values
    y_test = df.iloc[train_size:, -1].values
    return x_train, x_test, y_train, y_test


def fit_logistic_regression(x_train, y_train):
    logr_pipe = make_pipeline(StandardScaler(), LogisticRegression())
    logr_pipe.fit(x_train, y_train)
    return logr_pipe


def fit_decision_tree(x_train, y_train):
    tree_random_state = 42
    decision_tree = DecisionTreeClassifier(random_state=tree_random_state)
    decision_tree.fit(x_train, y_train)
    return decision_tree


def calculate_metrics(model, x_test, y_test):
    # Calculate predicted probabilities
    prob = model.predict_proba(x_test)[::, 1]
    # Calculate the ROC curve points
    fpr, tpr, _ = roc_curve(y_test, prob)
    # Calculate the AUC
    auc_score = roc_auc_score(y_test, prob)
    return prob, fpr, tpr, auc_score


import plotly.graph_objs as go


def plot_roc_curve(logr_fpr, logr_tpr, logr_auc, rf_fpr, rf_tpr, rf_auc, dtree_fpr, dtree_tpr, dtree_auc, ada_fpr, ada_tpr, ada_auc, xg_fpr, xg_tpr, xg_auc):
    # Create the figure
    fig = go.Figure()

    # Add the ROC curve for Logistic Regression
    fig.add_trace(
        go.Scatter(
            x=logr_fpr,
            y=logr_tpr,
            name=f"Logistic Regression (AUC={round(logr_auc, 6)})",
        )
    )

    # Add the diagonal line for random guess
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            line=dict(dash="dash"),
            mode="lines",
            showlegend=False,
        )
    )

    # Add the ROC curve for Random Forest
    fig.add_trace(
        go.Scatter(
            x=rf_fpr,
            y=rf_tpr,
            name=f"Random Forest (AUC={round(rf_auc, 6)})",
        )
    )

    # Add the ROC curve for Decision Tree
    fig.add_trace(
        go.Scatter(
            x=dtree_fpr,
            y=dtree_tpr,
            name=f"Decision Tree (AUC={round(dtree_auc, 6)})",
        )
    )

    # Add the ROC curve for AdaBoost
    fig.add_trace(
        go.Scatter(
            x=ada_fpr,
            y=ada_tpr,
            name=f"Ada Boost (AUC={round(ada_auc, 6)})",
        )
    )

    # Add the ROC curve for XGBoost
    fig.add_trace(
        go.Scatter(
            x=xg_fpr,
            y=xg_tpr,
            name=f"XG Boost (AUC={round(xg_auc, 6)})",
        )
    )

    # Update layout
    fig.update_layout(
        title="Receiver Operator Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        legend=dict(x=0.02, y=0.98)
    )

    return fig

def main():
    # Load data
    data_name = "Baseball"
    query = "SELECT * FROM features_table"
    df = maria_db(query, db_host="mariadb:3306")

    # Preprocess data
    df = preprocess_data(df)

    # Select features
    X, y = select_features(df)

    # Plots directory path
    this_dir = os.path.dirname(os.path.realpath(__file__))
    plot_dir = f"{this_dir}/Output/Plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Creating Predictors report as html
    report(df, X.columns, y.name, plot_dir, data_name)

    data_set = df_dict[data_set_nm.strip().lower()][0]
    predictors = df_dict[data_set_nm.strip().lower()][1]
    response = df_dict[data_set_nm.strip().lower()][2]

    # Stats model
    logr_model_fitted = fit_stats_model(df[select_features()], df["HTWins"])

    # Split data
    x_train, x_test, y_train, y_test = split_data(df, select_features(), train_size)

    # Logistic Regression
    logr_pipe = fit_logistic_regression(x_train, y_train)

    # Decision Tree Classifier
    decision_tree = fit_decision_tree(x_train, y_train)


    # Define models to be used
    models = {
        "LogisticRegression": LogisticRegression(random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42)
        "RandomForest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": GradientBoostingClassifier(random_state=42)
    }

    # Train and evaluate models
    for name, model in models.items():
        model.fit(x_train, y_train)
        logr_res = model_results("LogisticReg", logr_pipe, x_test, y_test, plot_dir)
        dtree_res = model_results("DecisionTree", decision_tree, x_test, y_test, plot_dir)
        rf_model = model_results(name, model, x_test, y_test, plot_dir)
        ada = model_results(name, model, x_test, y_test, plot_dir)
        xg = model_results(name, model, x_test, y_test, plot_dir)

    # Determine response variable type
    if data_set[response].nunique() > 2:
        resp_type = "Cont"

    elif data_set[response].nunique() == 2:
        resp_type = "Bool"
    else:
        print("No.of categories response variable have?")

    # Determine predictor variable types
    pred_dict = {}
    for pred in predictors:
        if is_numeric_dtype(data_set[pred]):
            pred_dict[pred] = "Continuous"
        else:
            pred_dict[pred] = "Categorical"

    cat_pred_list = [pred for pred, typ in pred_dict.items() if typ == "Categorical"]
    cont_pred_list = [pred for pred, typ in pred_dict.items() if typ == "Continuous"]

    # Process continuous predictor variables
    cont_fet_prop_list = []
    for i in cont_pred_list:
        if resp_type == "Cont":
            dict1 = pg.cont_resp_cont_pred(data_set, i, response)
            dict2 = mp.morp_cont_resp_cont_pred(data_set, i, "Cont", response)
            dict3 = lin_reg_plots(data_set[response], data_set[i], i)
            plot_link2 = None
        else:
            dict1 = pg.cat_resp_cont_pred(data_set, i, response)
            dict2 = mp.morp_cat_resp_cont_pred(data_set, i, "Cont", response)
            dict3 = log_reg_plots(data_set[response], data_set[i], i)
            plot_link2 = dict1["plot_link_2"]

        cont_fet_prop_list.append(
            {
                "Feature_nm": i,
                "Plot_link1": dict1["plot_link_1"],
                "Plot_link2": plot_link2,
                "Weigh_morp": dict2["weigh_morp"],
                "Unweigh_morp": dict2["unweigh_morp"],
                "Morp_plot_link": dict2["Plot_link"],
                "P_value": dict3["P_value"],
                "T_value": dict3["T_value"],
            }
        )

    cont_fet_prop_df = (
        pd.DataFrame(cont_fet_prop_list)
        .sort_values(by="Weigh_morp", ascending=False)
        .reset_index(drop=True)
    )

    fig = px.bar(cont_fet_prop_df, x="Feature_nm", y="Weigh_morp")
    fig.show()

    # Process categorical predictor variables
    cat_fet_prop_list = []

    for i in cat_pred_list:
        if resp_type == "Cont":
            dict1 = pg.cont_resp_cat_pred(data_set, i, response)
            dict2 = mp.morp_cont_resp_cat_pred(data_set, i, "Cont", response)
        else:
            dict1 = pg.cat_resp_cat_pred(data_set, i, response)
            dict2 = mp.morp_cat_resp

        cat_fet_prop_dict = {
            "Feature_nm": i,
            "Weighted_morp": dict2["weighted_morp"],
            "Unweighted_morp": dict,
            "Morp_plot_link": dict2["plot_link"],
            "Plot_link1": dict1["plot_link_1"],
            "Plot_link2": dict1.get("plot_link_2"),
            "Chi_square": dict1.get("chi_square"),
            "P_value": dict1.get("p_value"),
        }

        cat_fet_prop_list.append(cat_fet_prop_dict)

    cat_fet_prop_df = pd.DataFrame(cat_fet_prop_list)

    if not cat_fet_prop_df.empty:
        cat_fet_prop_df = cat_fet_prop_df.sort_values(
            by="Weighted_morp", ascending=False
        ).reset_index(drop=True)
        fig = px.bar(
            cat_fet_prop_df,
            x="Feature_nm",
            y="Weighted_morp",
            title="Categorical Feature Importance",
        )
        fig.show()

    # cont_cont_correlation
    cont_cont_list = []
    for i, j in itertools.combinations(cont_pred_list, 2):
        cont_cont_dict = (
            {
                "Cont_1": i,
                "Cont_2": j,
                "Correlation": pc.cont_pred_corr(data_set[i], data_set[j]),
                "Cont_1_morp_url": mp.morp_cont_resp_cont_pred(
                    data_set, i, resp_type, response
                )["Plot_link"],
                "Cont_2_morp_url": mp.morp_cont_resp_cont_pred(
                    data_set, j, resp_type, response
                )["Plot_link"],
            }
            if resp_type == "Continuous"
            else {
                "Cont_1": i,
                "Cont_2": j,
                "Correlation": pc.cont_pred_corr(data_set[i], data_set[j]),
                "Cont_1_morp_url": mp.morp_cat_resp_cont_pred(
                    data_set, i, resp_type, response
                )["Plot_link"],
                "Cont_2_morp_url": mp.morp_cat_resp_cont_pred(
                    data_set, j, resp_type, response
                )["Plot_link"],
            }
        )
        cont_cont_list.append(cont_cont_dict)

    if len(cont_cont_list) >= 1:
        cont_cont_corr_df = pd.DataFrame(cont_cont_list)
        cont_cont_corr_df = cont_cont_corr_df.sort_values(
            by="Correlation", ascending=False
        ).reset_index(drop=True)
        cont_cont_corr_df = cont_cont_corr_df[
            cont_cont_corr_df["Cont_1"] != cont_cont_corr_df["Cont_2"]
            ]
        cont_cont_corr_html_plt = pc.heatmap_plot_corr(
            cont_cont_corr_df, "Cont_1", "Cont_2", "Corr"
        )
    else:
        cont_cont_corr_df = pd.DataFrame(cont_cont_list)

    # cat_cat_correlation

    cat_cat_list_t = []
    cat_cat_list_v = []
    for i in cat_pred_list:
        for j in cat_pred_list:
            if resp_type == "Continuous":
                corr_v = pc.cat_corr(data_set[i], data_set[j])
                corr_t = pc.cat_corr(data_set[i], data_set[j], tschuprow=True)
            elif resp_type == "Boolean":
                corr_v = pc.cat_corr(data_set[i], data_set[j])
                corr_t = pc.cat_corr(data_set[i], data_set[j], tschuprow=True)
            cat_cat_dict_t = {"Cat_1": i, "Cat_2": j, "Correlation_T": corr_t}
            cat_cat_dict_v = {"Cat_1": i, "Cat_2": j, "Correlation_V": corr_v}
            cat_cat_list_t.append(cat_cat_dict_t)
            cat_cat_list_v.append(cat_cat_dict_v)

    cat_cat_corr_t_df = pd.DataFrame(cat_cat_list_t)
    cat_cat_corr_v_df = pd.DataFrame(cat_cat_list_v)

    if len(cat_cat_corr_t_df) >= 1 and len(cat_cat_corr_v_df) >= 1:
        cat_cat_corr_t_df = cat_cat_corr_t_df[
            cat_cat_corr_t_df["Cat_1"] != cat_cat_corr_t_df["Cat_2"]
            ]
        cat_cat_corr_v_df = cat_cat_corr_v_df[
            cat_cat_corr_v_df["Cat_1"] != cat_cat_corr_v_df["Cat_2"]
            ]
        cat_cat_corr_t_html_plt = px.imshow(
            cat_cat_corr_t_df.pivot("Cat_1", "Cat_2", "Correlation_T").values,
            x=cat_cat_corr_t_df["Cat_1"].unique(),
            y=cat_cat_corr_t_df["Cat_2"].unique(),
            labels=dict(x="Category 1", y="Category 2", color="Correlation"),
        )
        cat_cat_corr_v_html_plt = px.imshow(
            cat_cat_corr_v_df.pivot("Cat_1", "Cat_2", "Correlation_V").values,
            x=cat_cat_corr_v_df["Cat_1"].unique(),
            y=cat_cat_corr_v_df["Cat_2"].unique(),
            labels=dict(x="Category 1", y="Category 2", color="Correlation"),
        )
    else:
        cat_cat_corr_t_html_plt = None
        cat_cat_corr_v_html_plt = None

    # cat_cont correlation
    cat_cont_list = [
        (
            a,
            b,
            pc.cat_cont_corr(data_set[a], data_set[b]),
            mp.morp_cont_resp_cat_pred(data_set, a, "Cat", response)["Plot_link"]
            if resp_type == "Cont"
            else mp.morp_cat_resp_cat_pred(data_set, a, "Cat", response)["Plot_link"],
            mp.morp_cont_resp_cont_pred(data_set, b, "Cont", response)["Plot_link"]
            if resp_type == "Cont"
            else mp.morp_cat_resp_cont_pred(data_set, b, "Cont", response)["Plot_link"],
        )
        for a in cat_pred_list
        for b in cont_pred_list
    ]

    if len(cat_cont_list) >= 1:
        cat_cont_corr_df = pd.DataFrame(
            cat_cont_list,
            columns=["Cat", "Cont", "Corr", "Cat_morp_url", "Cont_morp_url"],
        )
        cat_cont_corr_df = cat_cont_corr_df.sort_values(
            by="Correlation", ascending=False
        ).reset_index(drop=True)
        cat_cont_corr_html_plt = pc.heatmap_plot_corr(
            cat_cont_corr_df, "Cont", "Cat", "Corr"
        )
        cat_cont_corr_df = cat_cont_corr_df[
            cat_cont_corr_df["Cont"] != cat_cont_corr_df["Cat"]
            ]
    else:
        cat_cont_corr_df = pd.DataFrame(cat_cont_list)

    # cat_cat bruteforce
    if len(cat_pred_list) > 1:
        cat_pairs = [
            (x, y)
            for i, x in enumerate(cat_pred_list)
            for j, y in enumerate(cat_pred_list)
            if i < j
        ]
        cat_cat_2d_morp_df = pd.DataFrame(cat_pairs, columns=["Cat_1", "Cat_2"])
        cat_cat_2d_morp_df[
            ["Weighted_morp", "Unweighted_morp"]
        ] = cat_cat_2d_morp_df.apply(
            lambda row: pd.Series(
                cat_cat_2d_morp(data_set, row["Cat_1"], row["Cat_2"], response)
            ),
            axis=1,
        )
        cat_cat_2d_morp_df = cat_cat_2d_morp_df.merge(
            cat_cat_corr_t_df, on=["Cat_1", "Cat_2"]
        )
        cat_cat_2d_morp_df = cat_cat_2d_morp_df.merge(
            cat_cat_corr_v_df, on=["Cat_1", "Cat_2", "Cat_1_morp_url", "Cat_2_morp_url"]
        )
        cat_cat_2d_morp_df["Correlation_T_Abs"] = cat_cat_2d_morp_df[
            "Correlation_T"
        ].abs()
        cat_cat_2d_morp_df["Correlation_V_Abs"] = cat_cat_2d_morp_df[
            "Correlation_V"
        ].abs()
        cat_cat_2d_morp_df = cat_cat_2d_morp_df.sort_values(
            "Weighted_morp", ascending=False
        ).reset_index(drop=True)
    else:
        cat_cat_2d_morp_df = pd.DataFrame([])

    # cat_cont_bruteforce
    if len(cat_pred_list) > 0 and len(cont_pred_list) > 0:

        cat_cont_2d_morp_list = [
            {"Cat": i, "Cont": j, **cat_cont_2d_morp(data_set, i, j, response)}
            for i in cat_pred_list
            for j in cont_pred_list
            if i != j
        ]
        # convert the list of dictionaries into a DataFrame
        cat_cont_2d_morp_df = pd.DataFrame(cat_cont_2d_morp_list)

        cat_cont_2d_morp_df = pd.merge(
            cat_cont_2d_morp_df, cat_cont_corr_df, on=["Cat", "Cont"], how="left"
        )
        # add a column with the absolute values of the correlation
        cat_cont_2d_morp_df["Correlation_Abs"] = cat_cont_2d_morp_df[
            "Correlation"
        ].abs()
        # sort the DataFrame by weighted_morp in descending order and reset the index
        cat_cont_2d_morp_df = cat_cont_2d_morp_df.sort_values(
            "Weighted_morp", ascending=False
        ).reset_index(drop=True)
    else:
        cat_cont_2d_morp_df = pd.DataFrame([])

    # cont_cont_bruteforce
    if len(cont_pred_list) > 1:
        cont_cont_2d_morp_df = pd.DataFrame(
            columns=[
                "Cont_1",
                "Cont_2",
                "Weighted_morp",
                "Unweighted_morp",
                "Plot_link",
            ]
        )
        for i, j in itertools.combinations(cont_pred_list, 2):
            dict1 = cont_cont_2d_morp(data_set, i, j, response)
            cont_cont_2d_morp_df = cont_cont_2d_morp_df.append(
                {
                    "Cont_1": i,
                    "Cont_2": j,
                    "Weighted_morp": dict1["Weighted_morp"],
                    "Unweighted_morp": dict1["Unweighted_morp"],
                    "Plot_link": dict1["Plot_link"],
                },
                ignore_index=True,
            )
        cont_cont_2d_morp_df = cont_cont_2d_morp_df.merge(
            cont_cont_corr_df, on=["Cont_1", "Cont_2"]
        )
        cont_cont_2d_morp_df["Correlation_Abs"] = cont_cont_2d_morp_df[
            "Correlation"
        ].abs()
        cont_cont_2d_morp_df = (
            cont_cont_2d_morp_df.sort_index(axis=1, ascending=True)
            .sort_values("Weighted_morp", ascending=False)
            .reset_index(drop=True)
        )
    else:
        cont_cont_2d_morp_df = pd.DataFrame([])

    # styling of table
    table_styles = [
        {
            "selector": "",
            "props": [("border-collapse", "collapse"), ("border", "2px solid #ddd")],
        },
        {
            "selector": "th",
            "props": [
                ("background-color", "#f2f2f2"),
                ("color", "#333"),
                ("font-size", "16px"),
                ("font-family", "Arial, Helvetica, sans-serif"),
                ("border", "2px solid #ddd"),
                ("padding", "8px"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("font-size", "14px"),
                ("font-family", "Arial, Helvetica, sans-serif"),
                ("border", "2px solid #ddd"),
                ("padding", "8px"),
            ],
        },
        {
            "selector": "tr:nth-child(even)",
            "props": [("background-color", "#f2f2f2")],
        },
        {
            "selector": "tr:hover",
            "props": [("background-color", "#ddd")],
        },
    ]

    # prop_dfs_clicakble
    cat_fet_prop_df = cat_fet_prop_df.style.format(
        {"Plot_link1": url_click, "Plot_link2": url_click, "Morp_plot_link": url_click}
    ).set_table_styles(table_styles)

    cont_fet_prop_df = cont_fet_prop_df.style.format(
        {"Plot_link1": url_click, "Plot_link2": url_click, "Morp_plot_link": url_click}
    ).set_table_styles(table_styles)

    # corr_dfs_clickable
    # making corr dfs clickable
    cont_cont_corr_df = (
        cont_cont_corr_df.style.format(
            {"Cont_1_morp_url": url_click, "Cont_2_morp_url": url_click}
        )
        .set_table_styles(table_styles)
        .hide_index()
    )

    cat_cat_corr_t_df = (
        cat_cat_corr_t_df.style.format(
            {"Cat_1_morp_url": url_click, "Cat_2_morp_url": url_click}
        )
        .set_table_styles(table_styles)
        .hide_index()
    )

    cat_cat_corr_v_df = (
        cat_cat_corr_v_df.style.format(
            {"Cat_1_morp_url": url_click, "Cat_2_morp_url": url_click}
        )
        .set_table_styles(table_styles)
        .hide_index()
    )

    cat_cont_corr_df = (
        cat_cont_corr_df.style.format(
            {"Cat_morp_url": url_click, "Cont_morp_url": url_click}
        )
        .set_table_styles(table_styles)
        .hide_index()
    )

    # bruteforce_dfs_clickable

    # create a list of data frames
    dfs = [cat_cat_2d_morp_df, cat_cont_2d_morp_df, cont_cont_2d_morp_df]

    # create a dictionary to store the models and their corresponding metrics
    models = {
        "Logistic Regression": logr_pipe,
        "Random Forest": rf_model,
        "Decision Tree": decision_tree,
        "AdaBoost": ada,
        "XG Boost": xg,

    }
    metrics = {}
    for name, model in models.items():
        prob, fpr, tpr, auc_score = calculate_metrics(model, x_test, y_test)
        metrics[name] = {
            "predicted_probabilities": prob,
            "fpr": fpr,
            "tpr": tpr,
            "auc_score": auc_score
        }

    fig = plot_roc_curve(logr_fpr, logr_tpr, logr_auc, rf_fpr, rf_tpr, rf_auc, dtree_fpr, dtree_tpr, dtree_auc, ada_fpr,
                         ada_tpr, ada_auc, xg_fpr, xg_tpr, xg_auc)
    fig.show()

    # iterate through the data frames and apply the style
    for df in dfs:
        df.style.format(
            {
                "Cat_1_morp_url": url_click,
                "Cat_2_morp_url": url_click,
                "Cont_1_morp_url": url_click,
                "Cont_2_morp_url": url_click,
                "Cat_morp_url": url_click,
                "Cont_morp_url": url_click,
                "Plot_link": url_click,
            }
        ).set_table_styles(table_styles)

        # write html page using f-strings and multi-line strings

        with open("data.html", "w") as out:
            out.write(
                f"""
                <h5>Continuous Predictors Properties</h5>
                {cont_fet_prop_df.to_html()}
                <br><br>
                <h5>Categorical Predictors Properties</h5>
                {cat_fet_prop_df.to_html()}
                <h5>Categorical/ Categorical Correlation</h5>
                <br><br>
                <h4>Correlation Tschuprow Matrix Heatmap</h4>
                {cat_cat_corr_t_html_plt}
                <br><br>
                <h4>Correlation Cramer's Matrix Heatmap</h4>
                {cat_cat_corr_v_html_plt}
                <br><br>
                <h4>Correlation Tschuprow Matrix</h4>
                {cat_cat_corr_t_df.to_html()}
                <br><br>
                <h4>Correlation Cramer's Matrix</h4>
                {cat_cat_corr_v_df.to_html()}
                <br><br>
                <h5>Categorical/ Continuous Correlation</h5>
                {cat_cont_corr_html_plt}
                <br><br>
                <h4>Categorical/ Continuous Correlation Matrix</h4>
                {cat_cont_corr_df.to_html()}
                <br><br>
                <h5>Continuous/ Continuous Correlation</h5>
                {cont_cont_corr_html_plt}
                <br><br>
                <h4>Continuous/ Continuous Correlation Matrix</h4>
                {cont_cont_corr_df.to_html()}
                <br><br>
                <h4>Categorical Categorical Brute force combination</h4>
                {cat_cat_2d_morp_df.to_html()}
                <br><br>
                <h4>Categorical Continuous Brute force combination</h4>
                {cat_cont_2d_morp_df.to_html()}
                <br><br>
                <h4>Continuous Continuous Brute force combination</h4>
                {cont_cont_2d_morp_df.to_html()}
            """
            )

    if __name__ == "__main__":
        sys.exit(main())
