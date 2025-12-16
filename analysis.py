import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


class Analysis:
    def __init__(self, betas_df):
        """Initialize Analysis class with betas dataframe."""
        self.betas = betas_df.copy()
        self.beta_cols = ['beta_12m', 'beta_24m', 'beta_36m']

    def descriptive_stats(self, x):
        """Calculate comprehensive descriptive statistics for a series."""
        x = x.dropna()
        if len(x) == 0:
            return pd.Series({
                'N': 0,
                'mean': np.nan,
                'std': np.nan,
                'skew': np.nan,
                'kurtosis': np.nan,
                'min': np.nan,
                '1%': np.nan,
                '5%': np.nan,
                '25%': np.nan,
                '50%': np.nan,
                '75%': np.nan,
                '95%': np.nan,
                '99%': np.nan,
                'max': np.nan
            })

        return pd.Series({
            'N': x.count(),
            'mean': x.mean(),
            'std': x.std(),
            'skew': skew(x) if len(x) > 2 else np.nan,
            'kurtosis': kurtosis(x) if len(x) > 3 else np.nan,
            'min': x.min(),
            '1%': np.percentile(x, 1),
            '5%': np.percentile(x, 5),
            '25%': np.percentile(x, 25),
            '50%': np.percentile(x, 50),
            '75%': np.percentile(x, 75),
            '95%': np.percentile(x, 95),
            '99%': np.percentile(x, 99),
            'max': x.max()
        })

    def get_descriptive_stats_by_industry(self, output_path=None):
        """Calculate descriptive statistics for beta by industry (not year)."""
        desc_stats = self.betas.groupby('industry')[self.beta_cols].apply(
            lambda df: df.apply(self.descriptive_stats)
        ).unstack()

        desc_stats = desc_stats.reindex(
            columns=['N', 'mean', 'std', 'skew', 'kurtosis', 'min',
                    '1%', '5%', '25%', '50%', '75%', '95%', '99%', 'max'],
            level=1
        )

        if output_path:
            desc_stats.to_csv(output_path)

        return desc_stats

    def get_annual_stats(self):
        """Calculate annual statistics (mean and std) by year and industry."""
        annual_stats = self.betas.groupby(['year', 'industry'])[self.beta_cols].agg(['mean', 'std']).reset_index()
        annual_stats.columns = ['year', 'industry',
                                'beta_12m_mean', 'beta_12m_std',
                                'beta_24m_mean', 'beta_24m_std',
                                'beta_36m_mean', 'beta_36m_std']
        return annual_stats

    def analyze_missing_betas(self):
        """Analyze missing beta values by year and industry."""
        missing_analysis = []

        for year in self.betas['year'].unique():
            for industry in self.betas['industry'].unique():
                subset = self.betas[(self.betas['year'] == year) & (self.betas['industry'] == industry)]
                total = len(subset)

                if total > 0:
                    for beta_col in self.beta_cols:
                        missing_count = subset[beta_col].isna().sum()
                        missing_pct = (missing_count / total) * 100
                        missing_analysis.append({
                            'year': year,
                            'industry': industry,
                            'beta_period': beta_col,
                            'total_obs': total,
                            'missing_count': missing_count,
                            'missing_pct': missing_pct
                        })

        return pd.DataFrame(missing_analysis)

    def get_volatility_trends(self):
        """Calculate annual average volatility components."""
        vol_cols = ["TVOL_12m", "SVOL_12m", "IVOL_12m",
                    "TVOL_24m", "SVOL_24m", "IVOL_24m",
                    "TVOL_36m", "SVOL_36m", "IVOL_36m"]

        vol_trends = self.betas.groupby("year")[vol_cols].mean().reset_index()
        return vol_trends

    def form_quintile_portfolios(self, sort_col, ret_col="excess_stock",
                                beta_col="beta_12m", mktcap_col="mktcap",
                                time_col="year"):
        """Form quintile portfolios sorted on a given column."""
        df = self.betas.copy()
        df[mktcap_col] = df[mktcap_col].abs()
        ew_list, vw_list = [], []

        for t, grp in df.groupby(time_col):
            grp = grp.dropna(subset=[sort_col]).copy()
            if grp.empty:
                continue

            grp["quintile"] = pd.qcut(grp[sort_col], 5, labels=False, duplicates="drop") + 1

            for q, qgrp in grp.groupby("quintile"):
                ew_ret = qgrp[ret_col].mean()
                ew_beta = qgrp[beta_col].mean()
                ew_list.append([t, q, ew_ret, ew_beta])

                weights = qgrp[mktcap_col] / qgrp[mktcap_col].sum()
                vw_ret = np.sum(weights * qgrp[ret_col])
                vw_beta = np.sum(weights * qgrp[beta_col])
                vw_list.append([t, q, vw_ret, vw_beta])

        ew_results = pd.DataFrame(ew_list, columns=[time_col, "quintile", "ew_ret", "ew_beta"])
        vw_results = pd.DataFrame(vw_list, columns=[time_col, "quintile", "vw_ret", "vw_beta"])

        return ew_results, vw_results

    def compute_spread(self, portfolios, ret_col):
        """Compute average spread between Q5 and Q1 portfolios."""
        def spread_for_year(g):
            q1 = g.loc[g["quintile"] == 1, ret_col]
            q5 = g.loc[g["quintile"] == 5, ret_col]
            if q1.empty or q5.empty:
                return np.nan
            return q5.values[0] - q1.values[0]

        spread = portfolios.groupby("year").apply(spread_for_year)
        return spread.mean(skipna=True)
