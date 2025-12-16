from data_processor import PreProcessor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

class FeatureEngineer():
        def __init__(self):
                self.sample = PreProcessor().get_data()[0]
                self.crsp = PreProcessor().get_data()[1]
                self.model = LinearRegression()
                
        def excess_returns(self):
                """Engineers features related to excess market return and excess stock return"""
                self.sample["excess_stock"] = self.sample['RETX'] - self.sample["rf"]
                self.sample["excess_mkt"] = self.sample["vwretd"] - self.sample["rf"]
        
        def compute_sampled_betas(self, lookback_periods=[12,24,36]):
                """Engineers rolling beta for each lookback period"""
                beta_results = []
                model = LinearRegression()
                for _, row in self.sample.iterrows():
                        permno = row["PERMNO"]
                        year = row["year"]
                        end_date = pd.to_datetime(f"{year}-12-31")
                        res = {"PERMNO": permno, "year": year}
                        for lb in lookback_periods:
                                start_date = end_date - pd.DateOffset(months=lb)
                                window = self.crsp[
                                        (self.crsp["PERMNO"] == permno) &
                                        (self.crsp["date"] > start_date) &
                                        (self.crsp["date"] <= end_date)
                                ]
                                mask = window[["excess_stock", "excess_mkt"]].notna().all(axis=1)
                                window = window.loc[mask]
                                if len(window) < 3:
                                        res[f"beta_{lb}m"] = np.nan
                                        continue
                                y = window["excess_stock"].values
                                X = window["excess_mkt"].values.reshape(-1, 1)
                                model.fit(X, y)
                                res[f"beta_{lb}m"] = model.coef_[0]
                                res[f"alpha_{lb}m"] = model.intercept_
                        beta_results.append(res)
                beta_df = pd.DataFrame(beta_results)
                merged = self.sample.merge(beta_df, on=["PERMNO", "year"], how="left")
                return merged

        def calculate_volatilities(self, betas, lookback_months=12):
                beta_col = f"beta_{lookback_months}m"
                tvol_col, svol_col, ivol_col = (
                        f"TVOL_{lookback_months}m",
                        f"SVOL_{lookback_months}m",
                        f"IVOL_{lookback_months}m"
                )

                out = []

                crsp = self.crsp.copy()
                crsp["date"] = pd.to_datetime(crsp["date"])
                betas = betas.copy()
                betas["date"] = pd.to_datetime(betas["date"])

                for idx, row in betas.iterrows():
                        permno, end_date, beta = row["PERMNO"], row["date"], row[beta_col]

                        if pd.isna(beta):
                                out.append([np.nan, np.nan, np.nan])
                                continue

                        start_date = end_date - pd.DateOffset(months=lookback_months)
                        window = crsp[(crsp["PERMNO"] == permno) &
                                (crsp["date"] > start_date) &
                                (crsp["date"] <= end_date)]

                        if len(window) < 3:
                                out.append([np.nan, np.nan, np.nan])
                                continue

                        var_stock = np.var(window["excess_stock"], ddof=1)
                        var_mkt = np.var(window["excess_mkt"], ddof=1)
                        residuals = window["excess_stock"] - beta * window["excess_mkt"]
                        var_resid = np.var(residuals, ddof=1)

                        tvol = np.sqrt(var_stock)
                        svol = beta * np.sqrt(var_mkt)
                        ivol = np.sqrt(var_resid)

                        out.append([tvol, svol, ivol])

                betas[[tvol_col, svol_col, ivol_col]] = pd.DataFrame(out, index=betas.index)

                return betas
