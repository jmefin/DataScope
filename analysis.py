import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

class AdvancedAnalysis:
    def __init__(self, db_manager):
        self.db_manager = db_manager

    def calculate_basic_stats(self, dataset_ids: list, metrics: list, filtered_data: pd.DataFrame = None) -> Dict:
        data = filtered_data if filtered_data is not None else self.db_manager.get_dataset_data(dataset_ids, metrics)
        stats_dict = {}
        
        for metric in metrics:
            if metric in data.columns:
                series = data[metric].dropna()
                stats_dict[metric] = {
                    'count': len(series),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    '25%': series.quantile(0.25),
                    'median': series.median(),
                    '75%': series.quantile(0.75),
                    'max': series.max(),
                    'range': series.max() - series.min(),
                    'cv%': (series.std() / series.mean() * 100) if series.mean() != 0 else np.nan
                }
        return stats_dict

    def calculate_correlations(self, dataset_ids: list, metrics: list) -> pd.DataFrame:
        data = self.db_manager.get_dataset_data(dataset_ids, metrics)
        return data[metrics].corr()

    def detect_anomalies(self, dataset_ids: list, metric: str, 
                        window: int = 20, threshold: float = 2.0) -> pd.DataFrame:
        data = self.db_manager.get_dataset_data(dataset_ids, [metric])
        
        data['rolling_mean'] = data[metric].rolling(window=window).mean()
        data['rolling_std'] = data[metric].rolling(window=window).std()
        data['zscore'] = np.abs((data[metric] - data['rolling_mean']) / data['rolling_std'])
        data['is_anomaly'] = data['zscore'] > threshold
        
        return data

    def trend_analysis(self, dataset_ids: list, metric: str) -> Dict:
        data = self.db_manager.get_dataset_data(dataset_ids, [metric])
        
        # Simple linear regression
        x = np.arange(len(data))
        y = data[metric].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }

    def cross_correlation(self, dataset_ids: list, 
                         metric1: str, metric2: str, 
                         max_lags: int = 24) -> Tuple[list, list]:
        data = self.db_manager.get_dataset_data(dataset_ids, [metric1, metric2])
        
        # Calculate cross-correlation for different lags
        lags = range(-max_lags, max_lags + 1)
        correlations = [data[metric1].corr(data[metric2].shift(lag)) for lag in lags]
        
        return lags, correlations
