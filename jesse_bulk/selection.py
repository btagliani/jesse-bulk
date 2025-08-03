"""
Advanced DNA Selection Engine for Jesse-Bulk

This module provides sophisticated multi-criteria selection algorithms
for choosing the best DNAs from optimization results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class MetricConfig:
    """Configuration for a single metric in the selection process"""
    metric: str
    weight: float
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    invert: bool = False  # True for metrics where lower is better
    normalization: str = "z-score"  # z-score, min-max, robust, percentile


@dataclass
class SelectionConfig:
    """Configuration for the entire selection strategy"""
    type: str  # composite_score, pareto, tournament
    metrics: List[MetricConfig]
    diversity_enabled: bool = False
    diversity_clusters: int = 5
    min_per_cluster: int = 1
    max_per_cluster: int = 3


class SelectionEngine:
    """
    Advanced DNA selection engine supporting multiple strategies
    and multi-criteria optimization.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the selection engine with configuration
        
        Args:
            config: Dictionary containing selection configuration
        """
        self.config = self._parse_config(config)
        self.scaler_cache = {}
        
    def _parse_config(self, config_dict: Dict) -> SelectionConfig:
        """Parse configuration dictionary into SelectionConfig object"""
        metrics = []
        for metric_cfg in config_dict.get('metrics', []):
            metrics.append(MetricConfig(
                metric=metric_cfg['metric'],
                weight=metric_cfg.get('weight', 1.0),
                min_threshold=metric_cfg.get('min_threshold'),
                max_threshold=metric_cfg.get('max_threshold'),
                invert=metric_cfg.get('invert', False),
                normalization=metric_cfg.get('normalization', 'z-score')
            ))
        
        diversity_cfg = config_dict.get('diversity', {})
        
        return SelectionConfig(
            type=config_dict.get('type', 'composite_score'),
            metrics=metrics,
            diversity_enabled=diversity_cfg.get('enabled', False),
            diversity_clusters=diversity_cfg.get('clusters', 5),
            min_per_cluster=diversity_cfg.get('min_per_cluster', 1),
            max_per_cluster=diversity_cfg.get('max_per_cluster', 3)
        )
    
    def select(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Select top N DNAs based on configured strategy
        
        Args:
            df: DataFrame with DNAs and metrics
            n: Number of DNAs to select
            
        Returns:
            DataFrame with selected DNAs
        """
        # Apply thresholds first
        filtered_df = self._apply_thresholds(df)
        
        if len(filtered_df) == 0:
            print("Warning: No DNAs passed the threshold criteria")
            return pd.DataFrame()
        
        # Apply selection strategy
        if self.config.type == 'composite_score':
            selected = self._composite_score_selection(filtered_df, n)
        elif self.config.type == 'pareto':
            selected = self._pareto_selection(filtered_df, n)
        elif self.config.type == 'tournament':
            selected = self._tournament_selection(filtered_df, n)
        else:
            raise ValueError(f"Unknown selection type: {self.config.type}")
        
        # Apply diversity if enabled
        if self.config.diversity_enabled and len(selected) > self.config.diversity_clusters:
            selected = self._apply_diversity(selected, n)
        
        return selected
    
    def _apply_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min/max thresholds to filter DNAs"""
        mask = pd.Series(True, index=df.index)
        
        for metric_cfg in self.config.metrics:
            if metric_cfg.metric not in df.columns:
                print(f"Warning: Metric '{metric_cfg.metric}' not found in data")
                continue
                
            if metric_cfg.min_threshold is not None:
                mask &= df[metric_cfg.metric] >= metric_cfg.min_threshold
                
            if metric_cfg.max_threshold is not None:
                mask &= df[metric_cfg.metric] <= metric_cfg.max_threshold
        
        filtered = df[mask]
        print(f"Threshold filtering: {len(df)} -> {len(filtered)} DNAs")
        
        return filtered
    
    def _normalize_metric(self, values: pd.Series, method: str, metric_name: str) -> pd.Series:
        """Normalize metric values using specified method"""
        # Handle missing values
        clean_values = values.dropna()
        
        if len(clean_values) == 0:
            return pd.Series(0, index=values.index)
        
        # Initialize result with zeros
        normalized = pd.Series(0, index=values.index)
        
        if method == 'z-score':
            if metric_name not in self.scaler_cache:
                self.scaler_cache[metric_name] = StandardScaler()
            scaler = self.scaler_cache[metric_name]
            normalized[clean_values.index] = scaler.fit_transform(
                clean_values.values.reshape(-1, 1)
            ).flatten()
            
        elif method == 'min-max':
            if metric_name not in self.scaler_cache:
                self.scaler_cache[metric_name] = MinMaxScaler()
            scaler = self.scaler_cache[metric_name]
            normalized[clean_values.index] = scaler.fit_transform(
                clean_values.values.reshape(-1, 1)
            ).flatten()
            
        elif method == 'robust':
            if metric_name not in self.scaler_cache:
                self.scaler_cache[metric_name] = RobustScaler()
            scaler = self.scaler_cache[metric_name]
            normalized[clean_values.index] = scaler.fit_transform(
                clean_values.values.reshape(-1, 1)
            ).flatten()
            
        elif method == 'percentile':
            # Rank-based normalization
            ranks = clean_values.rank(pct=True)
            normalized[clean_values.index] = ranks
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def _composite_score_selection(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Select DNAs using weighted composite scoring"""
        scores = pd.Series(0.0, index=df.index)
        
        total_weight = sum(m.weight for m in self.config.metrics if m.metric in df.columns)
        
        for metric_cfg in self.config.metrics:
            if metric_cfg.metric not in df.columns:
                continue
            
            # Normalize metric
            normalized = self._normalize_metric(
                df[metric_cfg.metric], 
                metric_cfg.normalization,
                metric_cfg.metric
            )
            
            # Invert if needed (for metrics where lower is better)
            if metric_cfg.invert:
                normalized = -normalized
            
            # Add weighted score
            weight_normalized = metric_cfg.weight / total_weight
            scores += normalized * weight_normalized
        
        # Add scores to dataframe and sort
        df_with_scores = df.copy()
        df_with_scores['composite_score'] = scores
        
        # Select top N
        selected = df_with_scores.nlargest(n, 'composite_score')
        
        print(f"Composite score selection: Top {len(selected)} DNAs")
        print(f"Score range: {selected['composite_score'].min():.3f} to {selected['composite_score'].max():.3f}")
        
        return selected
    
    def _pareto_selection(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Select DNAs using Pareto frontier (multi-objective optimization)"""
        # Get metrics for Pareto calculation
        metrics_data = []
        metric_names = []
        
        for metric_cfg in self.config.metrics:
            if metric_cfg.metric in df.columns:
                values = df[metric_cfg.metric].values
                if metric_cfg.invert:
                    values = -values  # Convert to maximization problem
                metrics_data.append(values)
                metric_names.append(metric_cfg.metric)
        
        if len(metrics_data) == 0:
            return df.head(n)
        
        metrics_array = np.column_stack(metrics_data)
        
        # Find Pareto frontier
        pareto_mask = self._is_pareto_efficient(metrics_array)
        pareto_df = df[pareto_mask]
        
        print(f"Found {len(pareto_df)} DNAs on Pareto frontier")
        
        # If we have more than needed, use composite scoring to rank within Pareto set
        if len(pareto_df) > n:
            return self._composite_score_selection(pareto_df, n)
        elif len(pareto_df) < n:
            # Add more from non-Pareto set using composite scoring
            non_pareto = df[~pareto_mask]
            additional = self._composite_score_selection(non_pareto, n - len(pareto_df))
            return pd.concat([pareto_df, additional])
        else:
            return pareto_df
    
    def _is_pareto_efficient(self, costs: np.ndarray) -> np.ndarray:
        """
        Find Pareto efficient points
        :param costs: An (n_points, n_costs) array
        :return: A boolean array of Pareto efficient points
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Remove dominated points
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] > c, axis=1
                )
                is_efficient[i] = True
        return is_efficient
    
    def _tournament_selection(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Select DNAs using tournament-style selection"""
        # For now, implement as composite score
        # TODO: Implement proper tournament rounds
        return self._composite_score_selection(df, n)
    
    def _apply_diversity(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Apply diversity-aware selection using clustering"""
        # Extract hyperparameters for clustering
        param_columns = [col for col in df.columns if col.startswith('params.')]
        
        if len(param_columns) == 0:
            print("Warning: No parameter columns found for diversity analysis")
            return df.head(n)
        
        # Prepare data for clustering
        param_data = df[param_columns].fillna(0)
        
        # Perform clustering
        n_clusters = min(self.config.diversity_clusters, len(df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(param_data)
        
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = clusters
        
        # Select from each cluster
        selected_dnas = []
        remaining_slots = n
        
        for cluster_id in range(n_clusters):
            cluster_df = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            # Determine how many to select from this cluster
            min_select = min(self.config.min_per_cluster, len(cluster_df), remaining_slots)
            max_select = min(self.config.max_per_cluster, len(cluster_df), remaining_slots)
            
            # Use composite score within cluster
            if 'composite_score' in cluster_df.columns:
                cluster_sorted = cluster_df.nlargest(max_select, 'composite_score')
            else:
                cluster_sorted = self._composite_score_selection(cluster_df, max_select)
            
            selected_dnas.append(cluster_sorted.head(max_select))
            remaining_slots -= len(cluster_sorted)
            
            if remaining_slots <= 0:
                break
        
        result = pd.concat(selected_dnas)
        print(f"Diversity selection: {len(result)} DNAs from {n_clusters} clusters")
        
        return result.head(n)
    
    def get_selection_report(self, df: pd.DataFrame, selected_df: pd.DataFrame) -> Dict:
        """Generate detailed report about the selection process"""
        report = {
            'total_candidates': len(df),
            'total_selected': len(selected_df),
            'selection_type': self.config.type,
            'metrics_used': [m.metric for m in self.config.metrics],
            'diversity_enabled': self.config.diversity_enabled
        }
        
        # Add metric statistics
        metric_stats = {}
        for metric_cfg in self.config.metrics:
            if metric_cfg.metric in selected_df.columns:
                metric_stats[metric_cfg.metric] = {
                    'min': selected_df[metric_cfg.metric].min(),
                    'max': selected_df[metric_cfg.metric].max(),
                    'mean': selected_df[metric_cfg.metric].mean(),
                    'std': selected_df[metric_cfg.metric].std()
                }
        
        report['metric_stats'] = metric_stats
        
        # Add diversity statistics if applicable
        if self.config.diversity_enabled and 'cluster' in selected_df.columns:
            cluster_counts = selected_df['cluster'].value_counts().to_dict()
            report['cluster_distribution'] = cluster_counts
        
        return report


# Preset configurations for common selection strategies
SELECTION_PRESETS = {
    'conservative': {
        'type': 'composite_score',
        'metrics': [
            {'metric': 'training_log.sharpe_ratio', 'weight': 0.4, 'min_threshold': 1.0},
            {'metric': 'training_log.win_rate', 'weight': 0.3, 'min_threshold': 0.55},
            {'metric': 'training_log.max_drawdown', 'weight': 0.3, 'max_threshold': 0.15, 'invert': True}
        ]
    },
    'aggressive': {
        'type': 'composite_score',
        'metrics': [
            {'metric': 'training_log.net_profit_percentage', 'weight': 0.5, 'min_threshold': 100},
            {'metric': 'training_log.expectancy_percentage', 'weight': 0.3, 'min_threshold': 1.0},
            {'metric': 'training_log.ratio_avg_win_loss', 'weight': 0.2, 'min_threshold': 1.5}
        ]
    },
    'balanced': {
        'type': 'composite_score',
        'metrics': [
            {'metric': 'training_log.sharpe_ratio', 'weight': 0.3, 'min_threshold': 0.8},
            {'metric': 'training_log.net_profit_percentage', 'weight': 0.3, 'min_threshold': 50},
            {'metric': 'training_log.win_rate', 'weight': 0.2, 'min_threshold': 0.5},
            {'metric': 'training_log.max_drawdown', 'weight': 0.2, 'max_threshold': 0.2, 'invert': True}
        ],
        'diversity': {
            'enabled': True,
            'clusters': 5,
            'min_per_cluster': 1,
            'max_per_cluster': 2
        }
    },
    'robust': {
        'type': 'pareto',
        'metrics': [
            {'metric': 'training_log.sharpe_ratio', 'weight': 1.0},
            {'metric': 'testing_log.sharpe_ratio', 'weight': 1.0},
            {'metric': 'training_log.max_drawdown', 'weight': 1.0, 'invert': True}
        ]
    }
}