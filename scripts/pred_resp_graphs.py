import path as p
import plotly.figure_factory as ff
import plotly.graph_objects as go


class PlotGraph:
    @staticmethod
    def cat_resp_cat_pred(data_set, pred_col, resp_col):
        pivoted_data = (
            data_set.groupby([resp_col, pred_col]).size().unstack(fill_value=0)
        )

        heatmap = go.Heatmap(
            x=pivoted_data.columns,
            y=pivoted_data.index,
            z=pivoted_data.values,
            colorscale="Blues",
        )

        # Define the layout of the plot
        layout = go.Layout(title="Heatmap", xaxis_title=pred_col, yaxis_title=resp_col)

        # Create the figure
        fig = go.Figure(data=[heatmap], layout=layout)

        file_name = f"{p.GLOBAL_PATH}/cat_{resp_col}_cat_{pred_col}_heatmap.html"

        # Show the plot
        fig.write_html(
            file=file_name,
            include_plotlyjs="cdn",
        )
        return {"column_nm": pred_col, "plot_link_1": file_name, "plot_link_2": None}

    @staticmethod
    def cat_resp_cont_pred(data_set, pred_col, resp_col):
        # Create histogram data
        hist_data = []
        group_labels = []
        for label, group in data_set.groupby(resp_col):
            hist_data.append(group[pred_col])
            group_labels.append(str(label))

        # Create distribution plot with custom bin_size
        fig_1 = ff.create_distplot(hist_data, group_labels)
        fig_1.update_layout(
            title=f"Categorical {resp_col} vs Continuous {pred_col}",
            xaxis_title=pred_col,
            yaxis_title=resp_col,
        )
        file_name_1 = f"{p.GLOBAL_PATH}/cat_{resp_col}_cont_{pred_col}_distplot.html"
        fig_1.write_html(file=file_name_1, include_plotlyjs="cdn")

        # Create violin plot
        fig_2 = go.Figure()
        for label, hist in zip(group_labels, hist_data):
            fig_2.add_trace(
                go.Violin(
                    x=[label] * len(hist),
                    y=hist,
                    name=label,
                    box_visible=True,
                    meanline_visible=True,
                )
            )
        fig_2.update_layout(
            title=f"Categorical {resp_col} vs Continuous {pred_col}",
            xaxis_title=resp_col,
            yaxis_title=pred_col,
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
        fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
        fig_1.update_layout(
            title=f"Continuous {resp_col} vs Categorical {pred_col}",
            xaxis_title=resp_col,
            yaxis_title="Distribution",
        )
        file_name_1 = f"{p.GLOBAL_PATH}/cont_{resp_col}_cat_{pred_col}_distplot.html"
        fig_1.write_html(file=file_name_1, include_plotlyjs="cdn")

        # Create violin plot
        fig_2 = go.Figure()
        for hist, label in zip(hist_data, group_labels):
            fig_2.add_trace(
                go.Violin(
                    x=[label] * len(data_set),
                    y=hist,
                    name=label,
                    box_visible=True,
                    meanline_visible=True,
                )
            )
        fig_2.update_layout(
            title=f"Continuous {resp_col} vs Categorical {pred_col}",
            xaxis_title="Groupings",
            yaxis_title=resp_col,
        )
        file_name_2 = f"{p.GLOBAL_PATH}/cont_{resp_col}_cat_{pred_col}_violin_plot.html"
        fig_2.write_html(file=file_name_2, include_plotlyjs="cdn")

        return {
            "column_nm": pred_col,
            "plot_link_1": file_name_1,
            "plot_link_2": file_name_2,
        }

    @staticmethod
    def cont_resp_cont_pred(data_set, pred_col, resp_col):
        # Create the scatter plot using plotly.graph_objects
        fig = go.Figure(
            data=go.Scatter(x=data_set[pred_col], y=data_set[resp_col], mode="markers")
        )

        # Add a trendline to the scatter plot
        fig.add_trace(
            go.Scatter(
                x=data_set[pred_col],
                y=fig["data"][0]["line"]["y"],
                mode="lines",
                line=dict(color="red"),
                name="trendline",
            )
        )

        # Add layout information
        fig.update_layout(
            title="Continuous " + resp_col + " vs " + " Continuous " + pred_col,
            xaxis_title=pred_col,
            yaxis_title=resp_col,
        )

        # Save the plot to a file
        file_name = (
            p.GLOBAL_PATH
            + "/cont_"
            + resp_col
            + "_cont_"
            + pred_col
            + "_scatter_plot.html"
        )
        fig.write_html(file=file_name, include_plotlyjs="cdn")

        # Return the result as a dictionary
        return {"column_nm": pred_col, "plot_link_1": file_name, "plot_link_2": None}
