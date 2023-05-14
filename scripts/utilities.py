from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import create_engine, text


# make_clickable function
def make_clickable(url: str) -> str:
    name = url.split("__")[-1].split(".")[0]
    return f'<a target="_blank" href="{url}">{name if len(name) <= 20 else "link"}</a>'


# mariadb_df function
def maria_db(
    query: str,
    db_user: str = "root",
    db_pass: str = "maria_dbr",  # pragma: allowlist secret
    db_host: str = "localhost",
    db_database: str = "baseball",
) -> DataFrame:
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )
    with create_engine(connect_string).connect() as conn:
        return pd.read_sql_query(query, conn)


# model_results function
def model_results(name, model, x_test, y_test, write_dir):
    predictions = model.predict(x_test)
    cr = classification_report(y_test, predictions, output_dict=True)
    print(f"\n{name} classication :\n{classification_report(y_test, predictions)}")
    mcc, mae, acc = (
        matthews_corrcoef(y_test, predictions),
        mean_absolute_error(y_test, predictions),
        accuracy_score(y_test, predictions),
    )
    pre, rec, f1, auc, cm = (
        precision_score(y_test, predictions, pos_label=1),
        recall_score(y_test, predictions, pos_label=1),
        f1_score(y_test, predictions, pos_label=1),
        roc_auc_score(y_test, predictions),
        confusion_matrix(y_test, predictions),
    )

    pd.DataFrame(cr).transpose().to_html(f"{write_dir}/{name}_cr.html")

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["HTLoses", "HTWins"],
            y=["HTLoses", "HTWins"],
            hoverongaps=False,
            texttemplate="%{z}",
        ),
        layout={
            "title": "<b> confuse_matrix </b>",
            "xaxis": {"title": "<b> predict)value </b>"},
            "yaxis": {"title": "<b> true_value </b>"},
            "height": 500,
            "paper_bgcolor": "#fafafa",
        },
    )

    fig.write_html(f"{write_dir}/{name}_cm.html")

    return [name, acc, auc, pre, rec, f1, mae, mcc]
