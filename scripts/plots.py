import numpy as np
import pandas as pd
import path as p
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class Plots:
    # morp=mean of response predictors
    @staticmethod
    def weigh_and_unweigh_morp(bin_mean_df, hist_df, df, x, y, resp_col):
        # Calculate unweighted mean of response predictors
        unweighted_morp = ((df[resp_col].mean() - bin_mean_df[resp_col]) ** 2).mean()

        # Calculate weighted mean of response predictors
        merged_df = pd.merge(bin_mean_df[[x, resp_col]], hist_df, on=x)
        merged_df["pop_prop"] = merged_df[y] / merged_df[y].sum()
        weighted_morp = (
            merged_df["pop_prop"] * (merged_df[resp_col] - df[resp_col].mean()) ** 2
        ).sum()

        return unweighted_morp, weighted_morp

    @staticmethod
    def morp_cat_resp_cat_pred(df, pred_col, pred_type, resp_col):
        # Calculate bin means for given class
        hist_df = df[pred_col].value_counts(sort=False).reset_index()
        bin_mean_df = (
            df.groupby(pred_col)[resp_col]
            .mean()
            .reset_index()
            .rename(columns={resp_col: f"{resp_col}_mean"})
        )

        # Create bar chart with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=hist_df["index"], y=hist_df[pred_col], name=pred_col),
            secondary_y=False,
        )
        # 1st graph
        fig.add_trace(
            go.Scatter(
                x=bin_mean_df[pred_col],
                y=bin_mean_df[f"{resp_col}_mean"],
                mode="lines+markers",
                name=f"bin_avg_for_{resp_col}",
                marker=dict(color="red"),
            ),
            secondary_y=True,
        )
        # 2nd graph
        fig.add_trace(
            go.Scatter(
                x=bin_mean_df[pred_col],
                y=[df[resp_col].mean()] * len(bin_mean_df),
                mode="lines",
                name=f"{pred_col}_avg",
            ),
            secondary_y=True,
        )
        # update layout
        fig.update_layout(
            title_text=f"Mean_response_plot_for_{pred_col}_and_{resp_col.lower()}",
            yaxis=dict(title="Total Population", side="left"),
            yaxis2=dict(
                title="Response",
                side="right",
                overlaying="y",
                tickmode="auto",
            ),
            xaxis=dict(title=f"{pred_col}_bins"),
        )

        # Save chart as HTML file
        file_name = f"{p.GLOBAL_PATH}/cat_{resp_col}_cat_{pred_col}.html"
        fig.write_html(file=file_name, include_plotlyjs="cdn")

        # Calculate MORP values
        weighted_morp, unweighted_morp = Plots.weigh_and_unweigh_morp(
            bin_mean_df, hist_df, df, pred_col, resp_col, resp_col
        )

        # Return dictionary
        return {
            "col_name": pred_col,
            "col_type": pred_type,
            "weighted_morp": weighted_morp,
            "unweighted_morp": unweighted_morp,
            "Plot_link": file_name,
        }

    @staticmethod
    def morp_cat_resp_cont_pred(df, pred_col, pred_type, resp_col):
        class_nm = "bool_true"
        # input column_nm and class_nm
        x = pred_col + "_bin"
        y = resp_col

        # creating bins for each column and calculating their average
        bins = pd.cut(df[pred_col], bins=10, right=True)
        df[x] = bins.mid
        bin_mean_df = df.groupby(x)[y].mean().reset_index()

        # creating the histogram of bins
        hist_df = bins.value_counts().reset_index()
        hist_df.columns = ["bin", x]

        # calculating morp values
        weighted_morp, unweighted_morp = Plots.weigh_and_unweigh_morp(
            bin_mean_df, hist_df, df, x, y, resp_col
        )

        # creating the plot
        fig_bar = make_subplots(specs=[[{"secondary_y": True}]])
        fig_bar.add_trace(
            go.Bar(x=hist_df["bin"], y=hist_df[x], name=pred_col), secondary_y=False
        )

        # adding the first graph
        fig_bar.add_trace(
            go.Scatter(
                x=bin_mean_df[x],
                y=bin_mean_df[y],
                name="bin_avg_for_" + class_nm,
                mode="lines+markers",
                marker=dict(color="Red"),
            ),
            secondary_y=True,
        )

        # adding the second graph
        fig_bar.add_trace(
            go.Scatter(
                x=bin_mean_df[x],
                y=[df[y].mean()] * len(bin_mean_df[x]),
                name=class_nm + "_avg",
                mode="lines",
            ),
            secondary_y=True,
        )

        # updating the layout
        fig_bar.update_layout(
            title_text="Mean_resp_plot_for_"
            + pred_col
            + "_and_"
            + (class_nm.lower()).replace("-", "_"),
            yaxis=dict(title=dict(text="Total_population"), side="left"),
            yaxis2=dict(
                title=dict(text="Resp"), side="right", overlaying="y", tickmode="auto"
            ),
            xaxis=dict(title=dict(text=pred_col + "_bins")),
        )

        # saving the plot to an HTML file
        file_name = (
            p.GLOBAL_PATH + "/" + "cat_" + resp_col + "_cont_" + pred_col + ".html"
        )
        fig_bar.write_html(file=file_name, include_plotlyjs="cdn")

        return {
            "col_name": pred_col,
            "col_type": pred_type,
            "weighted_morp": weighted_morp,
            "unweighted_morp": unweighted_morp,
            "Plot_link": file_name,
        }

    @staticmethod
    def morp_cont_resp_cat_pred(df, pred_colmn, pred_type, resp_col):
        x = pred_colmn
        y = resp_col

        # Find bin mean for given class
        hist_df = df[x].value_counts().sort_index().reset_index(name=x)
        bin_mean_df = df.groupby(x)[y].mean().reset_index().sort_values(x)

        # Create bar chart with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # first subgraph of bins and their counts
        fig.add_trace(
            go.Bar(x=hist_df["index"], y=hist_df[x], name=x), secondary_y=False
        )
        # Second subgraph of bin mean for given class
        fig.add_trace(
            go.Scatter(
                x=bin_mean_df[x],
                y=bin_mean_df[y],
                name=f"bin_avg_for_{resp_col}",
                mode="lines+markers",
                marker=dict(color="red"),
            ),
            secondary_y=True,
        )
        # overall avg graph for given class
        fig.add_trace(
            go.Scatter(
                x=bin_mean_df[x],
                y=[df[y].mean()] * len(bin_mean_df),
                name=f"{pred_colmn}_avg",
                mode="lines",
            ),
            secondary_y=True,
        )
        # updating layout
        fig.update_layout(
            title_text=f"Mean_response_plot_for_{pred_colmn}_and_{resp_col.lower()}",
            yaxis=dict(title="Total Population", side="left"),
            yaxis2=dict(
                title="Response", side="right", overlaying="y", tickmode="auto"
            ),
            xaxis=dict(title=f"{pred_colmn}_bins"),
        )
        # Save chart as HTML file
        file_name = f"{p.GLOBAL_PATH}/cont_{resp_col}_cat_{pred_colmn}.html"
        fig.write_html(file=file_name, include_plotlyjs="cdn")

        # Calculate MORP values
        weighted_morp, unweighted_morp = Plots.weighted_and_unweighted_morp(
            bin_mean_df, hist_df, df, x, y, resp_col
        )

        return {
            "col_name": pred_colmn,
            "col_type": pred_type,
            "weighted_morp": weighted_morp,
            "unweighted_morp": unweighted_morp,
            "Plot_link": file_name,
        }

    @staticmethod
    def morp_cont_resp_cont_pred(df, pred_col, pred_type, resp_col):
        # create a new column with bin labels
        n_bins = 10
        bin_edges = pd.cut(df[pred_col], bins=n_bins, right=True).unique().sort_values()
        bin_labels = np.arange(n_bins)
        df[f"{pred_col}_bin"] = pd.cut(df[pred_col], bins=bin_edges, labels=bin_labels)

        # compute the bin counts and mean response values
        bin_counts = df[f"{pred_col}_bin"].value_counts().sort_index()
        bin_means = df.groupby(f"{pred_col}_bin")[resp_col].mean().values

        # compute MORP values
        diff_bin_means = np.abs(np.diff(bin_means))
        unweighted_morp = np.mean(diff_bin_means)
        weighted_morp = np.average(diff_bin_means, weights=bin_counts[:-1])

        # create the plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=bin_labels, y=bin_counts, name=pred_col, secondary_y=False)
        fig.add_scatter(
            x=bin_labels,
            y=bin_means,
            name=f"bin_avg_for_{resp_col}",
            mode="lines+markers",
            marker=dict(color="Red"),
            secondary_y=True,
        )
        fig.add_scatter(
            x=bin_labels,
            y=[df[resp_col].mean()] * n_bins,
            name=f"{pred_col}_avg",
            mode="lines",
            secondary_y=True,
        )
        fig.update_layout(
            title_text=f"Mean_resp_plot_for_{pred_col}_and_{resp_col.lower()}",
            xaxis_title_text=f"{pred_col}_bins",
            yaxis=dict(title="Total_Population", side="left"),
            yaxis2=dict(title="Resp", side="right", overlaying="y", tickmode="auto"),
        )

        # save the plot to an HTML file
        file_name = f"{p.GLOBAL_PATH}/cont_{resp_col}_cont_{pred_col}.html"
        fig.write_html(file=file_name, include_plotlyjs="cdn")

        # return a dictionary with MORP values and plot link
        return {
            "col_name": pred_col,
            "col_type": pred_type,
            "weighted_morp": weighted_morp,
            "unweighted_morp": unweighted_morp,
            "Plot_link": file_name,
        }
