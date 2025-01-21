import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

class AdvancedAnalysis:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.analysis_time_range = ('08:00', '17:00')  # Default time range
        self.exclude_weekends = True  # Default to excluding weekends
        
    def set_analysis_time_range(self, start_time: str, end_time: str):
        """Set the time range for analysis (format: 'HH:MM')"""
        self.analysis_time_range = (start_time, end_time)
        
    def set_exclude_weekends(self, exclude: bool):
        """Set whether to exclude weekends from analysis"""
        self.exclude_weekends = exclude
        
    def _filter_by_time_range(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data by the current analysis time range and weekend exclusion"""
        if 'timestamp' not in data.columns:
            return data
            
        # Convert timestamp column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
        # Apply weekend exclusion if enabled
        if self.exclude_weekends:
            data = data[data['timestamp'].dt.weekday < 5]  # 0-4 = Monday-Friday
            
        # Skip time range filtering if analysis_time_range is None
        if self.analysis_time_range is None or None in self.analysis_time_range:
            return data
            
        # Convert time strings to time objects
        start_time = pd.to_datetime(self.analysis_time_range[0]).time()
        end_time = pd.to_datetime(self.analysis_time_range[1]).time()
        
        # Filter by time range
        time_mask = (data['timestamp'].dt.time >= start_time) & \
                   (data['timestamp'].dt.time <= end_time)
        return data[time_mask]

    def calculate_basic_stats(self, dataset_ids: list, metrics: list, filtered_data: pd.DataFrame = None) -> Dict:
        data = filtered_data if filtered_data is not None else self.db_manager.get_dataset_data(dataset_ids, metrics)
        data = self._filter_by_time_range(data)
        stats_dict = {}
        
        # Identify PM metrics
        pm_metrics = [m for m in metrics if m.startswith('pm')]
        other_metrics = [m for m in metrics if not m.startswith('pm')]
        
        # Forward fill PM metrics
        if pm_metrics:
            data[pm_metrics] = data[pm_metrics].ffill()
        
        for metric in metrics:
            if metric in data.columns:
                series = data[metric].dropna()
                
                # Use last value for PM metrics in range statistics
                if metric in pm_metrics:
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
                        'cv%': (series.std() / series.mean() * 100) if series.mean() != 0 else np.nan,
                        'last': series.iloc[-1] if len(series) > 0 else np.nan
                    }
                else:
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

    def calculate_range_statistics(self, dataset_ids: list, metric: str) -> Dict:
        """Calculate percentage of values in optimal, warning and alarm ranges"""
        data = self.db_manager.get_dataset_data(dataset_ids, [metric])
        thresholds = self.db_manager.get_metric_thresholds(metric)
        
        if not thresholds:
            return None
            
        # Get the metric values
        values = data[metric].dropna()
        if len(values) == 0:
            return None
            
        # Calculate ranges
        optimal_mask = (values >= thresholds['optimal_min']) & \
                      (values <= thresholds['optimal_max'])
                      
        warning_mask = ((values >= thresholds['alarm_min']) & 
                       (values < thresholds['optimal_min'])) | \
                      ((values > thresholds['optimal_max']) & 
                       (values <= thresholds['alarm_max']))
                       
        alarm_mask = (values < thresholds['alarm_min']) | \
                    (values > thresholds['alarm_max'])
        
        return {
            'optimal_pct': optimal_mask.mean() * 100,
            'warning_pct': warning_mask.mean() * 100,
            'alarm_pct': alarm_mask.mean() * 100,
            'total_values': len(values)
        }

    def cross_correlation(self, dataset_ids: list, 
                         metric1: str, metric2: str, 
                         max_lags: int = 24) -> Tuple[list, list]:
        data = self.db_manager.get_dataset_data(dataset_ids, [metric1, metric2])
        
        # Calculate cross-correlation for different lags
        lags = range(-max_lags, max_lags + 1)
        correlations = [data[metric1].corr(data[metric2].shift(lag)) for lag in lags]
        
        return lags, correlations


class ParticleAnalysis(AdvancedAnalysis):
    def __init__(self, db_manager):
        super().__init__(db_manager)
        
    def calculate_pm_average(self, dataset_ids: list, pm_metrics: list, filtered_data: pd.DataFrame = None) -> Dict[str, float]:
        """Calculate average PM values in mg/m³ for selected metrics"""
        if not pm_metrics:
            return {}
            
        # Use provided filtered data or get new data
        if filtered_data is None:
            data = self.db_manager.get_dataset_data(dataset_ids, pm_metrics)
            filtered_data = self._filter_by_time_range(data)
        
        # Forward fill PM values
        filtered_data[pm_metrics] = filtered_data[pm_metrics].ffill()
        
        # Calculate averages in mg/m³
        averages = {}
        total_sum = 0.0
        valid_metrics = 0
        
        for metric in pm_metrics:
            if metric in filtered_data.columns:
                series = filtered_data[metric].dropna()
                if len(series) > 0:
                    # Convert from µg/m³ to mg/m³
                    avg = series.mean() / 1000
                    averages[metric] = avg
                    total_sum += avg
                    valid_metrics += 1
                    
        # Calculate total average if we have valid metrics
        if valid_metrics > 0:
            averages['total_average'] = total_sum / valid_metrics
            
        return averages
