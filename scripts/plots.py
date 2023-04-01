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
        # Create a pivot table with the mean response for each category
        pivot_table = pd.pivot_table(
            df, values=resp_col, index=pred_col, aggfunc=np.mean
        ).reset_index()

        # Create a bar chart with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # figure1
        fig.add_trace(
            go.Bar(x=df[pred_col].value_counts().index, y=df[pred_col].value_counts().values, name=pred_col),
            secondary_y=False,
        )
        # figure2
        fig.add_trace(
            go.Scatter(
                x=pivot_table[pred_col],
                y=pivot_table[resp_col],
                mode="lines+markers",
                name=f"bin_avg_for_{resp_col}",
                marker=dict(color="red"),
            ),
            secondary_y=True,
        )

        # Add layout information
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

        # Save the plot to a file
        file_name = f"{p.GLOBAL_PATH}/cat_{resp_col}_cat_{pred_col}.html"
        fig.write_html(file=file_name, include_plotlyjs="cdn")

        # Calculate MORP values
        hist_counts = df[pred_col].value_counts(sort=False).values
        weighted_morp = np.sum(hist_counts * np.abs(pivot_table[resp_col] - df[resp_col].mean()))
        unweighted_morp = np.abs(pivot_table[resp_col].diff().mean())

        # Return the result as a dictionary
        return {
            "col_name": pred_col,
            "col_type": pred_type,
            "weighted_morp": weighted_morp,
            "unweighted_morp": unweighted_morp,
            "Plot_link": file_name,
        }

    @staticmethod
    def morp_cat_resp_cont_pred(df, pred_col, pred_type, resp_col):
        # Define binning column name
        bin_col = f"{pred_col}_bin"

        # Bin the predictor variable and calculate mean response for each bin
        df[bin_col] = pd.cut(df[pred_col], bins=10, right=True).apply(lambda x: x.mid)
        bin_mean_df = df.groupby(bin_col)[resp_col].mean().reset_index()

        # Create histogram of bins
        hist_df = pd.DataFrame(df[bin_col].value_counts(sort=False))
        hist_df.index.name = bin_col
        hist_df.columns = ["Total Population"]

        # Calculate MORP values
        weighted_morp, unweighted_morp = Plots.weigh_and_unweigh_morp(
            bin_mean_df, hist_df, df, bin_col, resp_col, resp_col
        )

        # Create plot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=hist_df.index, y=hist_df["Total Population"], name="Total Population"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=bin_mean_df[bin_col],
                y=bin_mean_df[resp_col],
                mode="lines+markers",
                name=f"bin_avg_for_{resp_col}",
                marker=dict(color="red"),
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=bin_mean_df[bin_col],
                y=[df[resp_col].mean()] * len(bin_mean_df),
                mode="lines",
                name=f"{bin_col}_avg",
            ),
            secondary_y=True,
        )
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
        file_name = f"{p.GLOBAL_PATH}/cat_{resp_col}_cont_{pred_col}.html"
        fig.write_html(file=file_name, include_plotlyjs="cdn")

        # Return dictionary
        return {
            "col_name": pred_col,
            "col_type": pred_type,
            "weighted_morp": weighted_morp,
            "unweighted_morp": unweighted_morp,
            "Plot_link": file_name,
        }

    @staticmethod
    def morp_cont_resp_cat_pred(df, pred_col, pred_type, resp_col):
        x = pred_col
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
                name=f"{pred_col}_avg",
                mode="lines",
            ),
            secondary_y=True,
        )
        # updating layout
        fig.update_layout(
            title_text=f"Mean_resp_plot_for_{pred_col}_and_{resp_col.lower()}",
            yaxis=dict(title="Total Population", side="left"),
            yaxis2=dict(
                title="Response", side="right", overlaying="y", tickmode="auto"
            ),
            xaxis=dict(title=f"{pred_col}_bins"),
        )
        # Save chart as HTML file
        file_name = f"{p.GLOBAL_PATH}/cont_{resp_col}_cat_{pred_col}.html"
        fig.write_html(file=file_name, include_plotlyjs="cdn")

        # Calculate MORP values
        weighted_morp, unweighted_morp = Plots.weighted_and_unweighted_morp(
            bin_mean_df, hist_df, df, x, y, resp_col
        )

        return {
            "col_name": pred_col,
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
