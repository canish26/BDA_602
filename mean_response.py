import numpy as np
import pandas as pd
import plotly.graph_objects as go

csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "Species"]

iris = pd.read_csv(csv_url, names=attributes)

# stats using numpy
def numpy_stats(array, column_list):

    mean = np.mean(array, axis=1)
    min = np.min(array, axis=1)
    max = np.max(array, axis=1)
    quant = np.quantile(array, [0.25, 0.75, 0.95], axis=0)
    out_arr = np.vstack((mean, min, max, quant))
    out_df = pd.DataFrame(
        data=out_arr,
        index=["mean", "min", "max", "quartile", "median", "third quartile"],
        columns=column_list,
    )
    return out_df


# mean response
iris_sepal_length = iris[["sepal_length", "Species"]]

a = np.array(iris_sepal_length["sepal_length"])
count, bins = np.histogram(a, bins=10, range=(np.min(a), np.max(a)))
bins_mod = 0.5 * (bins[:-1] + bins[1:])

iris_sepal_length_setosa = iris_sepal_length.loc[
    iris_sepal_length["Species"] == "Iris-setosa"
]
b = np.array(iris_sepal_length_setosa["sepal_length"])
count_Iris_Setosa, _ = np.histogram(b, bins=bins)

count_response = count_Iris_Setosa / count

species_class_response_rate = len(iris.loc[iris["Species"] == "Iris-setosa"]) / len(
    iris
)
species_class_response_rate_arr = np.array(
    [species_class_response_rate] * len(bins_mod)
)

print(species_class_response_rate_arr)


fig = go.Figure(
    data=go.Bar(
        x=bins_mod,
        y=count,
        name="sepal length",
        marker=dict(color="blue"),
    )
)

fig.add_trace(
    go.Scatter(
        x=bins_mod,
        y=count_response,
        yaxis="y2",
        name="Response",
        marker=dict(color="red"),
    )
)


fig.add_trace(
    go.Scatter(
        x=bins_mod,
        y=species_class_response_rate_arr,
        yaxis="y2",
        mode="lines",
        name="Species 3",
    )
)


fig.update_layout(
    title_text="Iris Setosa vs Sepal length Mean of Response Rate Plot",
    legend=dict(orientation="v"),
    yaxis=dict(
        title=dict(text="Total Population"),
        side="left",
        range=[0, 30],
    ),
    yaxis2=dict(
        title=dict(text="Response"),
        side="right",
        range=[-0.1, 1.2],
        overlaying="y",
        tickmode="auto",
    ),
)
fig.write_html(file="Diff Plot Iris Setosa.html", include_plotlyjs="cdn")
# Set x-axis title
fig.update_xaxes(title_text="Predictor Bins")
