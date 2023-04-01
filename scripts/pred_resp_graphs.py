import path as p
import plotly.express as px
# import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.subplots as sp


class PlotGraph:
    @staticmethod
    def cat_resp_cat_pred(data_set, pred_col, resp_col):
        fig = px.imshow(
            data_set.groupby([resp_col, pred_col]).size().unstack(fill_value=0),
            x=pred_col,
            y=resp_col,
            color_continuous_scale="Blues",
        )

        file_name = f"{p.GLOBAL_PATH}/cat_{resp_col}_cat_{pred_col}_heatmap.html"

        # Show the plot
        fig.write_html(
            file=file_name,
            include_plotlyjs="cdn",
        )

        return {"column_nm": pred_col, "plot_link_1": file_name, "plot_link_2": None}

    @staticmethod
    def cat_resp_cont_pred(data_set, pred_col, resp_col):
        fig_1 = px.histogram(
            data_set,
            x=pred_col,
            color=resp_col,
            marginal="rug",
            histnorm="density",
            nbins=20,
        )

        fig_1.update_layout(
            title=f"Categorical {resp_col} vs Continuous {pred_col}",
            xaxis_title=pred_col,
            yaxis_title="Density",
        )

        file_name_1 = f"{p.GLOBAL_PATH}/cat_{resp_col}_cont_{pred_col}_distplot.html"
        fig_1.write_html(file=file_name_1, include_plotlyjs="cdn")

        fig_2 = px.violin(
            data_set,
            x=resp_col,
            y=pred_col,
            box=True,
            points="all",
            color=resp_col,
            title=f"Categorical {resp_col} vs Continuous {pred_col}",
            yaxis_title=pred_col,
            xaxis_title=resp_col,
        )

        file_name_2 = f"{p.GLOBAL_PATH}/cat_{resp_col}_cont_{pred_col}_violin_plot.html"
        fig_2.write_html(file=file_name_2, include_plotlyjs="cdn")

        return {
            "column_nm": pred_col,
            "plot_link_1": file_name_1,
            "plot_link_2": file_name_2,
        }

    @staticmethod
    def cont_resp_cat_pred(data_set, pred_col, resp_col):
        # Group data together and create labels
        hist_data = [
            data_set.loc[data_set[pred_col] == group, resp_col]
            for group in data_set[pred_col].unique()
        ]
        group_labels = data_set[pred_col].unique()

        # Create distribution plot with custom bin_size
        dist_figs = []
        for i, hist in enumerate(hist_data):
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=hist, nbinsx=50, name=group_labels[i]))
            fig.update_layout(
                title=f"{group_labels[i]} Distribution of {resp_col}",
                xaxis_title=resp_col,
                yaxis_title="Frequency",
                height=400,
                width=400,
            )
            dist_figs.append(fig)

        # Create violin plot
        violin_figs = []
        for i, hist in enumerate(hist_data):
            fig = go.Figure()
            fig.add_trace(
                go.Violin(
                    x=[group_labels[i]] * len(hist),
                    y=hist,
                    name=group_labels[i],
                    box_visible=True,
                    meanline_visible=True,
                )
            )
            fig.update_layout(
                title=f"{group_labels[i]} {resp_col} vs {pred_col}",
                xaxis_title=pred_col,
                yaxis_title=resp_col,
                height=400,
                width=400,
            )
            violin_figs.append(fig)

        # Combine subplots
        fig = sp.make_subplots(
            rows=2, cols=len(group_labels), subplot_titles=tuple(group_labels)
        )

        for i, subplot in enumerate(dist_figs):
            fig.add_trace(subplot["data"][0], row=1, col=i + 1)
        for i, subplot in enumerate(violin_figs):
            fig.add_trace(subplot["data"][0], row=2, col=i + 1)

        fig.update_layout(
            title=f"Continuous {resp_col} vs Categorical {pred_col}",
            height=800,
            width=1000,
        )

        file_name = f"{p.GLOBAL_PATH}/cont_{resp_col}_cat_{pred_col}_subplots.html"
        fig.write_html(file=file_name, include_plotlyjs="cdn")

        return {
            "column_nm": pred_col,
            "plot_link": file_name,
        }

    @staticmethod
    def cont_resp_cont_pred(data_set, pred_col, resp_col):
        # Create scatter plot with trendline
        fig = px.scatter(data_set, x=pred_col, y=resp_col, trendline="ols")

        # Add layout information
        fig.update_layout(
            title=f"Continuous {resp_col} vs Continuous {pred_col}",
            xaxis_title=pred_col,
            yaxis_title=resp_col,
        )

        # Save the plot to a file
        file_name = f"{p.GLOBAL_PATH}/cont_{resp_col}_cont_{pred_col}_scatter_plot.html"
        fig.write_html(file=file_name, include_plotlyjs="cdn")

        # Return the result as a dictionary
        return {"column_nm": pred_col, "plot_link_1": file_name, "plot_link_2": None}
