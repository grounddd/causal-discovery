"""
Factor Filtering Module
Handles statistical filtering and selection of important factors
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set


class FactorFilter:
    def __init__(self, lift_threshold: float = 1.2, coverage_threshold: float = 0.05):
        """
        Initialize factor filter
        
        Args:
            lift_threshold: Minimum lift value for factor significance
            coverage_threshold: Minimum coverage ratio for factor relevance
        """
        self.lift_threshold = lift_threshold
        self.coverage_threshold = coverage_threshold
        self.factor_scores = {}
        
    def _compute_entropy(self, p: float) -> float:
        """Compute binary entropy"""
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    def compute_factor_scores(self, data: pd.DataFrame, target_col: str = 'is_hot') -> Dict:
        """
        Compute lift and coverage scores for all factors
        
        Args:
            data: Input dataframe
            target_col: Target column name
            
        Returns:
            Dictionary with factor scores
        """
        factor_scores = {}
        p_target = data[target_col].mean()
        
        for factor in data.columns:
            if factor == target_col:
                continue
                
            factor_values = data[factor].unique()
            max_lift = 0
            best_value = None
            
            for value in factor_values:
                if pd.isna(value):
                    continue
                    
                p_target_given_value = data[data[factor] == value][target_col].mean()
                
                if p_target > 0:
                    lift = p_target_given_value / p_target
                    if lift > max_lift:
                        max_lift = lift
                        best_value = value
            
            coverage = data[factor].notna().sum() / len(data)
            
            factor_scores[factor] = {
                'lift': max_lift,
                'coverage': coverage,
                'best_value': best_value,
                'events_with_factor': data[factor].notna().sum(),
                'hot_events_with_factor': data[data[factor].notna()][target_col].sum()
            }
        
        self.factor_scores = factor_scores
        return factor_scores
    
    def filter_factors(self, 
                      data: pd.DataFrame, 
                      target_factor_count: int = 27,
                      exclude_columns: List[str] = None) -> Set[str]:
        """
        Filter factors based on statistical criteria
        
        Args:
            data: Input dataframe
            target_factor_count: Number of factors to keep
            exclude_columns: Columns to exclude
            
        Returns:
            Set of filtered factor names
        """
        if exclude_columns is None:
            exclude_columns = [
                'is_hot', 'image_content_category', 'image_visual_information_density',
                'image_narrative_completeness', 'image_popular_elements', 'image_visual_attractiveness',
                'image_emotional_impact', 'image_symbolic_elements', 'image_authenticity_perception',
                'image_shock_value', 'has_opposing_sides'
            ]
        
        # Step 1: Exclude specified columns
        all_features = [col for col in data.columns if col not in exclude_columns]
        print(f"Excluded {len(exclude_columns)} factors, remaining: {len(all_features)}")
        
        # Step 2: Compute scores for remaining factors
        self.compute_factor_scores(data)
        
        # Step 3: Filter by statistical criteria
        filtered_factors = self._filter_by_criteria(all_features)
        
        # Step 4: Select top factors
        top_factors = self._select_top_factors(filtered_factors, target_factor_count)
        
        return top_factors
    
    def _filter_by_criteria(self, factors: List[str]) -> Dict[str, Dict]:
        """
        Filter factors based on lift and coverage thresholds
        """
        filtered = {}
        for factor in factors:
            if factor in self.factor_scores:
                scores = self.factor_scores[factor]
                if (scores['lift'] >= self.lift_threshold and 
                    scores['coverage'] >= self.coverage_threshold):
                    filtered[factor] = scores
        
        print(f"Statistical filtering: {len(factors)} -> {len(filtered)} factors")
        return filtered
    
    def _select_top_factors(self, filtered_factors: Dict[str, Dict], target_count: int) -> Set[str]:
        """
        Select top factors based on combined score
        """
        if len(filtered_factors) <= target_count:
            return set(filtered_factors.keys())
        
        # Compute combined score
        factor_scores = {}
        for factor, scores in filtered_factors.items():
            combined_score = scores['lift'] * scores['coverage']
            factor_scores[factor] = combined_score
        
        # Sort by combined score
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        top_factors = [factor for factor, _ in sorted_factors[:target_count]]
        
        print(f"Selected top {len(top_factors)} factors from {len(filtered_factors)} filtered factors")
        return set(top_factors)
    
    def save_filtered_data(self, 
                          original_data: pd.DataFrame, 
                          filtered_factors: Set[str],
                          output_path: str = 'filtered_data.csv') -> pd.DataFrame:
        """
        Save filtered dataset
        
        Args:
            original_data: Original dataframe
            filtered_factors: Set of factors to keep
            output_path: Path to save filtered data
            
        Returns:
            Filtered dataframe
        """
        filtered_data = original_data[list(filtered_factors) + ['is_hot']].copy()
        filtered_data.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"Filtered data saved to {output_path}")
        print(f"Shape: {filtered_data.shape}")
        print(f"Factors: {list(filtered_factors)}")
        
        return filtered_data