import numpy as np
import pandas as pd

# from scipy.stats import pearsonr
import plotly.express as px
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder


class Correlation_Preds:
    def fill_na(data):
        return (
            pd.Series(data).fillna(0).values
            if isinstance(data, pd.Series)
            else np.nan_to_num(data, nan=0)
        )

    def cont_pred_corr(df1, df2):
        return np.corrcoef(df1.fill_na(df1), df1.fill_na(df2))[0, 1]


class correlation_ratio:
    @staticmethod
    def cat_correlation(x, y, measure="pmi"):
        """
        Calculates correlation statistic for categorical-categorical association using either PMI or chi2 measures.

        Parameters:
        -----------
        x : list / ndarray / Pandas Series
            A sequence of categorical measurements
        y : list / NumPy ndarray / Pandas Series
            A sequence of categorical measurements
        measure : str, default = 'pmi'
            The correlation measure to be used. Possible values: 'pmi', 'chi2'

        Returns:
        --------
        float in the range of [-1,1]
        """
        le_x = LabelEncoder()
        le_y = LabelEncoder()
        x = le_x.fit_transform(x)
        y = le_y.fit_transform(y)
        crosstab_matrix = pd.crosstab(x, y)
        p_xy = crosstab_matrix / len(x)
        p_x = crosstab_matrix.sum(axis=1) / len(x)
        p_y = crosstab_matrix.sum(axis=0) / len(y)
        if measure == "pmi":
            pmi = np.log2(p_xy / (np.outer(p_x, p_y) + 1e-10))
            pmi[~np.isfinite(pmi)] = 0
            return np.mean(pmi)
        elif measure == "chi2":
            chi2, _, _, _ = chi2_contingency(crosstab_matrix)
            n = np.sum(crosstab_matrix)
            phi2 = chi2 / n
            r, c = crosstab_matrix.shape
            phi2corr = max(0, phi2 - ((r - 1) * (c - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            ccorr = c - ((c - 1) ** 2) / (n - 1)
            corr = np.sqrt(phi2corr / min((rcorr - 1), (ccorr - 1)))
            if phi2corr == 0:
                return 0
            else:
                return np.sign(corr) * np.sqrt(phi2corr)
        else:
            raise ValueError(
                "Invalid measure parameter. Possible values: 'pmi', 'chi2'"
            )


def corr_heatmap_plot(df, pred_col_1, pred_col_2, value_col):
    fig_heatmap = px.imshow(
        df.corr()[[pred_col_1, pred_col_2]].loc[[pred_col_1, pred_col_2]].values,
        x=[pred_col_1, pred_col_2],
        y=[pred_col_1, pred_col_2],
        color_continuous_scale="YlOrRd",
        zmin=-1,
        zmax=1,
        labels=dict(color=value_col),
    )

    fig_heatmap.update_layout(title="Correlation Heatmap")
    return fig_heatmap.show()
