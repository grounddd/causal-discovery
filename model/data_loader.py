"""
Data Loading and Preprocessing Module
Handles data loading, cleaning, and preparation
"""

import pandas as pd
import json
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self):
        """Initialize data loader"""
        self.data = None
        self.col2invmap = None
    
    def load_weibo_data(self, csv_file_path: str) -> tuple:
        """
        Load Weibo data from CSV file
        
        Args:
            csv_file_path: Path to CSV file
            
        Returns:
            tuple: (X, y, full_data)
        """
        df = pd.read_csv(csv_file_path)
        
        # Define columns to exclude
        exclude_columns = [
            'title', 'duration_hours', 'max_search_count', 'max_read_count',
            'max_discuss_count', 'max_interact_count', 'max_original_count'
        ]
        
        # Select feature columns
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        X = df[feature_columns[:-1]]  # Exclude is_hot
        y = df['is_hot']
        
        return X, y, df
    
    def load_filtered_data(self, filtered_csv_path: str = 'filtered_data.csv') -> tuple:
        """
        Load filtered dataset
        
        Args:
            filtered_csv_path: Path to filtered data CSV
            
        Returns:
            tuple: (X, y, filtered_data)
        """
        try:
            filtered_data = pd.read_csv(filtered_csv_path)
            
            # Separate features and target
            feature_columns = [col for col in filtered_data.columns if col != 'is_hot']
            X = filtered_data[feature_columns]
            y = filtered_data['is_hot']
            
            print(f"Loaded filtered data: {filtered_data.shape}")
            print(f"Features: {len(feature_columns)}")
            print(f"Samples: {len(filtered_data)}")
            print(f"Hot event ratio: {y.mean():.3f}")
            
            return X, y, filtered_data
            
        except FileNotFoundError:
            print(f"Filtered data file not found: {filtered_csv_path}")
            return None, None, None
            
        except Exception as e:
            print(f"Error loading filtered data: {e}")
            return None, None, None
    
    def load_inverse_mapping(self, json_path: str = 'col2invmap.json') -> dict:
        """
        Load inverse mapping from JSON file
        
        Args:
            json_path: Path to mapping JSON file
            
        Returns:
            Dictionary with inverse mappings
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.col2invmap = json.load(f)
            return self.col2invmap
            
        except FileNotFoundError:
            print(f"Mapping file not found: {json_path}")
            return {}
            
        except json.JSONDecodeError as e:
            print(f"Error parsing mapping file: {e}")
            return {}
    
    def decode_chain(self, path: list, col2invmap: dict) -> list:
        """
        Decode chain using inverse mapping
        
        Args:
            path: Chain path list
            col2invmap: Inverse mapping dictionary
            
        Returns:
            Decoded chain path
        """
        decoded = []
        
        for node in path:
            if '=' in node:
                col, val = node.split('=')
                
                if col in col2invmap:
                    try:
                        val_key = str(int(float(val)))
                        val = col2invmap[col].get(val_key, val)
                    except Exception:
                        pass  # Keep original value if decoding fails
                
                decoded.append(f"{col}={val}")
            else:
                decoded.append(node)
        
        return decoded
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> tuple:
        """
        Split data into train and test sets
        
        Args:
            X: Features
            y: Target
            test_size: Test set proportion
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=y
        )