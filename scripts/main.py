import os
# import sys

import path as p
import numpy as np
import pandas as pd
import plotly.express as px
from plots import Plots as mp
from pred_resp_graphs import Plot_Graph as pg
import statsmodels.api as sm
from dataset_loader import Test_Dataset
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

path = p.GLOBAL_PATH
os.makedirs(path, exist_ok=True)

path_2d_morp = p.GLOBAL_PATH_2D_MORP
os.makedirs(os.path.join(path_2d_morp), exist_ok=True)


def rf_ranking_cont_resp(df, cont_var_pred_list, cat_var_pred_list, resp):
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
            "Column_type": np.where(
                df.columns.isin(x_cont.columns), "Continuous", "Categorical"
            ),
            "fet_imp_coeff": rf.feature_importances_,
        }
    )

    return feature_importance.sort_values("fet_imp_coeff", ascending=False).reset_index(
        drop=True
    )


def rf_ranking_cat_resp(df, cont_var_pred_list, cat_var_pred_list, resp):
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
            "Column_type": "Continuous" if i in x_cont.columns else "Categorical",
            "fet_imp_coeff": j,
        }
        for i, j in zip(df, rf.feature_importances_)
    ]

    rank_list_df = pd.DataFrame(rank_list)
    return rank_list_df.sort_values("fet_imp_coeff", ascending=False).reset_index(
        drop=True
    )


def linear_reg_plots(y, x, fet_nm):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()

    # Get the stats
    t_value = round(model.tvalues[1], 6)
    p_value = "{:.6e}".format(model.pvalues[1])
    p_value = np.float64(p_value)

    return {
        "Column_name": fet_nm,
        "Column_type": "Continuous",
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
        "Column_type": "Continuous",
        "P_value": p_value,
        "T_value": t_value,
    }


def cat_cat_2d_morp(df_ip, x1, x2, y):
    df = df_ip.groupby([x1, x2])[y].agg(["mean", "size"]).reset_index()
    df.columns = [x1, x2, "mean", "size"]

    df["unweighted_morp"] = (df[y]["mean"].mean() - df["mean"]) ** 2
    df["weighted_morp"] = (df["size"] / df["size"].sum()) * df["unweighted_morp"]

    mean_size = df.apply(lambda row: f"{row['mean']:.6f} pop:{row['size']}", axis=1)

    fig_heatmap = px.imshow(
        df.pivot(index=x1, columns=x2, values="mean"),
        x=df[x2].unique(),
        y=df[x1].unique(),
        color_continuous_scale="YlGnBu",
        labels={
            "x": x2.replace("_bin", ""),
            "y": x1.replace("_bin", ""),
            "color": "Correlation",
        },
        title=f"{x2.replace('_bin', '')} vs {x1.replace('_bin', '')}",
    )

    fig_heatmap.update_traces(
        text=mean_size.values,
        texttemplate="%{text}",
        hovertemplate="%{x}<br>%{y}<br>Correlation: %{z:.6f}<br>Population: %{text}",
    )

    fig_heatmap.write_html(
        file=f"{path_2d_morp}/cat_{x1}_cat_{x2}_2D_morp.html", include_plotlyjs="cdn"
    )

    return {
        "Weighted_morp": df["weighted_morp"].sum(),
        "Unweighted_morp": df["unweighted_morp"].sum()
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
        labels=dict(
            x=x1.replace("_bin", ""), y=x2.replace("_bin", ""), color="Correlation"
        ),
        title=f"{x2.replace('_bin', '')} vs {x1.replace('_bin', '')}",
    )
    fig.update_traces(text=mean_size.values, texttemplate="%{text}")
    fig.write_html(
        file=f"{path_2d_morp}/cat_{x1}_cont_{x2_bin}_2D_morp.html",
        include_plotlyjs="cdn",
    )

    return {
        "Weighted_morp": weighted_morp.sum(),
        "Unweighted_morp": unweighted_morp.sum() / len(df_grouped),
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
    df["unweighted_morp"] = (df_ip[y].mean() - y_mean) ** 2
    df["weighted_morp"] = (df[y + "size"] / df[y + "size"].sum()) * df[
        "unweighted_morp"
    ]

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
        "Weighted_morp": df["weighted_morp"].sum(),
        "Unweighted_morp": df["unweighted_morp"].sum() / len(df),
        "Plot_link": file_name,
    }


def pred_typ(data_set, pred_list):
    return {
        i: "Categorical"
        if type(data_set[i][0]) == str
           or data_set[i].nunique() == 2
           and not data_set[i].dtype.kind in "iufc"
        else "Continuous"
        for i in pred_list
    }


def url_click(url):
    return (
        f'<a target="_blank" href="{url.split(",")[1] if "," in url else url}">plots link</a>'
        if url
        else ""
    )


def main():
    # get all available datasets
    test_datasets = Test_Dataset()
    available_datasets = test_datasets.get_all_available_datasets()

    # select a dataset
    selected_dataset = None
    while selected_dataset not in available_datasets:
        print("Please select one of the five datasets given below:")
        print("\n".join(available_datasets))
        selected_dataset = input().strip().lower()
        if selected_dataset not in available_datasets:
            print("Invalid dataset selected.")

    # load data for the selected dataset
    df, predictors, response = test_datasets.get_test_data_set(
        data_set_name=selected_dataset
    )

    def analyze_continuous_predictors(df, response, predictors):
        # Determine the type of response variable
        n_unique = len(df[response].unique())
        if n_unique > 2:
            response_type = "Continuous"
        elif n_unique == 2:
            response_type = "Boolean"
        else:
            raise ValueError(
                f"The response variable '{response}' has {n_unique} unique values."
            )

        # Determine the type of each predictor variable and analyze continuous predictors
        cont_fet_prop_list = []
        for pred_name, pred_type in pred_typ(df, predictors).items():
            if pred_type != "Continuous":
                continue

            # Perform analysis based on response type
            if response_type == "Continuous":
                # Perform continuous-response continuous-predictor analysis
                dict1 = pg.cont_resp_cont_pred(df, pred_name, response)
                dict2 = mp.morp_cont_resp_cont_pred(
                    df, pred_name, "Continuous", response
                )
                dict3 = linear_reg_plots(df[response], df[pred_name])
                plot_link2 = None
            else:  # Boolean response
                # Perform boolean-response continuous-predictor analysis
                dict1 = pg.cat_resp_cont_pred(df, pred_name, response)
                dict2 = mp.morp_cat_resp_cont_pred(
                    df, pred_name, "Continuous", response
                )
                dict3 = log_reg_plots(df[response], df[pred_name])
                plot_link2 = dict1["plot_link_2"]

            # Add feature properties to the list
            cont_fet_prop_list.append(
                {
                    "Feature_nm": pred_name,
                    "Plot_link1": dict1["plot_link_1"],
                    "Plot_link2": plot_link2,
                    "Weighted_morp": dict2["weighted_morp"],
                    "Unweighted_morp": dict2["unweighted_morp"],
                    "Morp_plot_link": dict2["Plot_link"],
                    "P_value": dict3["P_value"],
                    "T_value": dict3["T_value"],
                }
            )

        # Sort continuous predictor results by weighted MORP score
        if cont_fet_prop_list:
            cont_fet_prop_df = pd.DataFrame(cont_fet_prop_list)
            cont_fet_prop_df = cont_fet_prop_df.sort_values(
                by="Weighted_morp", ascending=False
            ).reset_index(drop=True)
            return cont_fet_prop_df
        else:
            print("No continuous predictors found.")
            return None


if __name__ == "__main__":
    main()
