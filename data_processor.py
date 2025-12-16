import pandas as pd
import numpy as np
import re
import yaml
from fredapi import Fred

SEED = 42

class PreProcessor():
    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize the PreProcessor with FRED API access."""
        
        with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                fred_api_key = config['api_keys']['fred']

        self.fred = Fred(api_key=fred_api_key)
        self.dtypes = {"PERMNO": "int64",
                    "SHRCD": "Int64",
                    "SICCD": "string",    
                    "TICKER": "string",
                    "COMNAM": "string",
                    "PERMCO": "Int64",
                    "CUSIP": "string",
                    "BIDLO": "float64",
                    "ASKHI": "float64",
                    "PRC": "float64",
                    "VOL": "float64",
                    "RET": "string",       
                    "BID": "float64",
                    "ASK": "float64",
                    "SHROUT": "float64",
                    "RETX": "string",      
                    "vwretd": "float64"
        }

    @staticmethod
    def clean_numeric(x):
        """Clean numeric values by removing non-numeric characters."""
        if pd.isna(x):
            return np.nan
        cleaned = re.sub(r'[^0-9eE\.\-+]', '', str(x))
        try:
            return float(cleaned)
        except:
            return np.nan

    @staticmethod
    def map_sic_to_industry(sic):
        if sic.isdigit():
            sic_int = int(sic)
            if 1 <= sic_int <= 999:
                return "Agriculture, Forestry and Fishing"
            elif 1000 <= sic_int <= 1499:
                return "Mining"
            elif 1500 <= sic_int <= 1799:
                return "Construction"
            elif 2000 <= sic_int <= 3999:
                return "Manufacturing"
            elif 4000 <= sic_int <= 4999:
                return "Transportation and other Utilities"
            elif 5000 <= sic_int <= 5199:
                return "Wholesale Trade"
            elif 5200 <= sic_int <= 5999:
                return "Retail Trade"
            elif 6000 <= sic_int <= 6799:
                return "Finance, Insurance and Real Estate"
            elif 7000 <= sic_int <= 8999:
                return "Services"
            elif 9000 <= sic_int <= 9999:
                return "Public Administration"
            else:
                return "Other"
        else:
            return "Other"
    
    def get_data(self, file_path = 'MSF_1996_2023.csv',sample_size: int = 10):
        """Load and process CRSP data with risk-free rate and industry classification."""
        # Read CRSP data into DataFrame
        crsp = pd.read_csv(file_path, dtype=self.dtypes, parse_dates=["date"]).dropna() # type: ignore
        
        # Clean values in 'RET' and 'RETX' columns
        crsp['RET'] = crsp['RET'].apply(self.clean_numeric)
        crsp['RETX'] = crsp['RETX'].apply(self.clean_numeric)

        # Read risk-free data using FRED API and merge into CRSP data
        rf = self.fred.get_series('DTB3')
        crsp['date'] = pd.to_datetime(crsp['date'])
        rf = rf.reset_index()
        rf.columns = ['date', 'rf']  
        rf['date'] = pd.to_datetime(rf['date'])
        crsp = crsp.merge(rf, on='date', how='left')
        crsp['rf'] = crsp['rf'].ffill()

        # Map SIC Codes to industry
        crsp['industry'] = crsp['SICCD'].apply(self.map_sic_to_industry)
        
        # Extract year and month
        crsp['year'] = crsp['date'].dt.year # type: ignore
        crsp['month'] = crsp['date'].dt.month # type: ignore

        # Create random sample of 10 companies per industry for each year
        sampled = (
            crsp
            .groupby(["year", "industry"], group_keys=False)
            .apply(lambda df: df.sample(n=min(sample_size, len(df)), random_state=SEED))
            .sort_values(["year", "industry", "PERMNO"])
        )

        return sampled, crsp