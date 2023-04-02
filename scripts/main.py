import os
import itertools
import path as p
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from plots import Plots as mp
from pred_resp_graphs import Plot_Graph as pg
import statsmodels.api as sm
from correlation_plots import Correlation_Preds as pc
from dataset_loader import Test_Dataset

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
            "Column_type": np.where(
                df.columns.isin(x_cont.columns), "Cont", "Cat"
            ),
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
            x=x1.replace("_bin", ""), y=x2.replace("_bin", ""), color="Corr"
        ),
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
    df["weigh_morp"] = (df[y + "size"] / df[y + "size"].sum()) * df[
        "unweigh_morp"
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
        "Weigh_morp": df["weigh_morp"].sum(),
        "Unweigh_morp": df["unweigh_morp"].sum() / len(df),
        "Plot_link": file_name,
    }


def main():
    df_dict = {}
    test_datasets = Test_Dataset()
    for test in test_datasets.get_all_available_datasets():
        df, predictors, response = test_datasets.get_test_data_set(data_set_name=test)
        df_dict[test] = [df, predictors, response]

    flag = True
    available_datasets = ["mpg", "tips", "titanic", "diabetes", "breast_cancer"]
    while flag:
        print("select one of the 5 datasets following:")
        print('\n'.join(available_datasets))
        data_set_nm = input().strip().lower()
        if data_set_nm in available_datasets:
            flag = False
        else:
            print("Not selected one of the above datasets")

    print("I have selected,", data_set_nm)

    data_set, predictors, response = df_dict[data_set_nm]

    # Determine response variable type
    if len(data_set[response].value_counts()) > 2:
        resp_type = "Cont"
    elif len(data_set[response].value_counts()) == 2:
        resp_type = "Bool"
    else:
        print("How many categories does response var got??")

    # Determine predictor variable types
    pred_dict = pred_typ(data_set, predictors)
    cat_pred_list = [i for i in pred_dict if pred_dict[i] == "Categorical"]
    cont_pred_list = [i for i in pred_dict if pred_dict[i] == "Continuous"]

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

        cont_fet_prop_list.append({
            "Feature_nm": i,
            "Plot_link1": dict1["plot_link_1"],
            "Plot_link2": plot_link2,
            "Weigh_morp": dict2["weigh_morp"],
            "Unweigh_morp": dict2["unweigh_morp"],
            "Morp_plot_link": dict2["Plot_link"],
            "P_value": dict3["P_value"],
            "T_value": dict3["T_value"],
        })

    cont_fet_prop_df = pd.DataFrame(cont_fet_prop_list).sort_values(
        by="Weigh_morp", ascending=False
    ).reset_index(drop=True)

    # Process categorical predictor variables
    cat_fet_prop_list = []
    for i in cat_pred_list:
        if resp_type == "Cont":
            dict1 = pg.cont_resp_cat_pred(data_set, i, response)
            dict2 = mp.morp_cont_resp_cat_pred(data_set, i, "Cont", response)
        else:
            dict1 = pg.cat_resp_cat_pred(data_set, i, response)
            dict2 = mp.morp_cat_resp

    # cont_cont_correlation
    cont_cont_list = []
    for a, b in itertools.combinations(cont_pred_list, 2):
        cont_cont_dict = {
            "Cont_a": a,
            "Cont_b": b,
            "Corr": pc.cont_pred_corr(data_set[a], data_set[b]),
            "Cont_a_morp_url": mp.morp_cont_resp_cont_pred(
                data_set, a, "Cont", response
            )["Plot_link"] if resp_type == "Cont" else mp.morp_cat_resp_cont_pred(
                data_set, a, "Cont", response
            )["Plot_link"],
            "Cont_b_morp_url": mp.morp_cont_resp_cont_pred(
                data_set, b, "Cont", response
            )["Plot_link"] if resp_type == "Cont" else mp.morp_cat_resp_cont_pred(
                data_set, b, "Cont", response
            )["Plot_link"],
        }
        cont_cont_list.append(cont_cont_dict)

    if cont_cont_list:
        cont_cont_corr_df = pd.DataFrame(cont_cont_list)
        cont_cont_corr_html_plt = pc.corr_heatmap_plots(
            cont_cont_corr_df, "Cont_a", "Cont_b", "Corr"
        )
        cont_cont_corr_df = cont_cont_corr_df.sort_values(
            by="Corr", ascending=False
        ).reset_index(drop=True)
        cont_cont_corr_df = cont_cont_corr_df[
            cont_cont_corr_df["Cont_a"] != cont_cont_corr_df["Cont_b"]
            ]
    else:
        cont_cont_corr_df = pd.DataFrame()

        # cat_cat_correlation
        cat_cat_list_t = [
            {
                "Cat_a": a,
                "Cat_b": b,
                "Corr_V": pc.cat_correlation(data_set[a], data_set[b]),
                "Cat_a_morp_url": mp.morp_cont_resp_cat_pred(data_set, a, "Cat", response)["Plot_link"],
                "Cat_b_morp_url": mp.morp_cont_resp_cat_pred(data_set, b, "Cat", response)["Plot_link"],
            } if resp_type == "Continuous" else {
                "Cat_a": a,
                "Cat_b": b,
                "Corr_V": pc.cat_correlation(data_set[a], data_set[b]),
                "Cat_a_morp_url": mp.morp_cat_resp_cat_pred(data_set, a, "Cat", response)["Plot_link"],
                "Cat_b_morp_url": mp.morp_cat_resp_cat_pred(data_set, b, "Cat", response)["Plot_link"],
            }
            for a in cat_pred_list
            for b in cat_pred_list
        ]

        cat_cat_list_t = [d for d in cat_cat_list_t if d["Cat_a"] != d["Cat_b"]]

        cat_cat_list_v = [
            {
                "Cat_1": a,
                "Cat_2": b,
                "Corr_T": pc.cat_correlation(data_set[a], data_set[b], tschuprow=True),
                "Cat_a_morp_url": mp.morp_cont_resp_cat_pred(data_set, a, "Cat", response)["Plot_link"],
                "Cat_b_morp_url": mp.morp_cont_resp_cat_pred(data_set, b, "Cat", response)["Plot_link"],
            } if resp_type == "Continuous" else {
                "Cat_a": a,
                "Cat_b": b,
                "Corr_T": pc.cat_correlation(data_set[a], data_set[b], tschuprow=True),
                "Cat_a_morp_url": mp.morp_cat_resp_cat_pred(data_set, a, "Cat", response)["Plot_link"],
                "Cat_b_morp_url": mp.morp_cat_resp_cat_pred(data_set, b, "Cat", response)["Plot_link"],
            }
            for a in cat_pred_list
            for b in cat_pred_list
        ]

        cat_cat_list_v = [d for d in cat_cat_list_v if d["Cat_a"] != d["Cat_b"]]

        cat_cat_corr_t_df = pd.DataFrame(cat_cat_list_t)
        cat_cat_corr_v_df = pd.DataFrame(cat_cat_list_v)

        cat_cat_corr_t_df = cat_cat_corr_t_df.sort_values(by="Corr_T", ascending=False).reset_index(drop=True)
        cat_cat_corr_v_df = cat_cat_corr_v_df.sort_values(by="Corr_V", ascending=False).reset_index(drop=True)

        cat_cat_corr_t_html_plt = pc.corr_heatmap_plots(cat_cat_corr_t_df, "Cat_a", "Cat_a", "Correlation_T")
        cat_cat_corr_v_html_plt = pc.corr_heatmap_plots(cat_cat_corr_v_df, "Cat_b", "Cat_b", "Correlation_V")

        # cat_cont correlation

        cat_cont_list = [(a, b, pc.cat_cont_correlation_ratio(data_set[a], data_set[b]),
                          mp.morp_cont_resp_cat_pred(data_set, a, "Cat", response)[
                              "Plot_link"] if resp_type == "Cont" else
                          mp.morp_cat_resp_cat_pred(data_set, a, "Cat", response)["Plot_link"],
                          mp.morp_cont_resp_cont_pred(data_set, b, "Cont", response)[
                              "Plot_link"] if resp_type == "Cont" else
                          mp.morp_cat_resp_cont_pred(data_set, b, "Cont", response)["Plot_link"]
                          )
                         for a in cat_pred_list
                         for b in cont_pred_list
                         ]

        if len(cat_cont_list) >= 1:
            cat_cont_corr_df = pd.DataFrame(cat_cont_list,
                                            columns=["Cat", "Cont", "Corr", "Cat_morp_url", "Cont_morp_url"])
            cat_cont_corr_df = cat_cont_corr_df.sort_values(by="Correlation", ascending=False).reset_index(drop=True)
            cat_cont_corr_html_plt = pc.corr_heatmap_plots(cat_cont_corr_df, "Cont", "Cat", "Corr")
            cat_cont_corr_df = cat_cont_corr_df[cat_cont_corr_df["Cont"] != cat_cont_corr_df["Cat"]]
        else:
            cat_cont_corr_df = pd.DataFrame(cat_cont_list)


if __name__ == "__main__":
    main()
