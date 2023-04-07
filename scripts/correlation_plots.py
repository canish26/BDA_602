import numpy as np
import pandas as pd
import warnings
from scipy import stats
import plotly.express as px
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder


class Correlation_Preds:
    @staticmethod
    def fill_na(data):
        if isinstance(data, pd.Series):
            return data.fillna(0).values
        else:
            return np.nan_to_num(data)

    @staticmethod
    def cont_pred_corr(df1, df2):
        return np.nan_to_num(np.corrcoef(df1, df2)[0, 1])

    @staticmethod
    def cat_cont_corr(x, y, measure="pmi"):
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

    @staticmethod
    def cat_corr(x, y, bias_correction=True, tschuprow=False):
        try:
            x, y = Correlation_Preds.fill_na(x), Correlation_Preds.fill_na(y)
            crosstab_matrix = pd.crosstab(x, y)
            chi2, _, _, _ = stats.chi2_contingency(crosstab_matrix, correction=crosstab_matrix.shape != (2, 2))
            phi2 = chi2 / crosstab_matrix.sum().sum()

            r, c = crosstab_matrix.shape
            n_observations = crosstab_matrix.sum().sum()
            r_corrected, c_corrected = r - ((r - 1) ** 2) / (n_observations - 1), c - ((c - 1) ** 2) / (
                        n_observations - 1)

            if bias_correction:
                phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
                denominator = np.sqrt((r_corrected - 1) * (c_corrected - 1))
            else:
                phi2_corrected, denominator = phi2, min((r_corrected - 1), (c_corrected - 1))

            corr_coeff = np.sqrt(np.nan_to_num(phi2_corrected / denominator))
            if tschuprow:
                corr_coeff = np.sqrt(np.nan_to_num(phi2_corrected / np.sqrt(denominator)))

            return corr_coeff
        except Exception as ex:
            print(ex)
            warnings.warn("Calculating error Tschuprow's T" if tschuprow else "Calculating error Cramer's V",
                          RuntimeWarning)
            return np.nan

    @staticmethod
    def heatmap_plot_corr(df, pred_col_1, pred_col_2, value_col):
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






