"""
Data preprocessing for injury detection ML models.
Loads XSens, insole, and eye tracker data and prepares it for feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class DataLoader:
    """Load and preprocess gait data from CSV files."""
    
    def __init__(self, data_dir: str = "src/data"):
        self.data_dir = Path(data_dir)
        
    def load_xsens(self, filename: str = "xsens.csv") -> pd.DataFrame:
        """
        Load XSens motion capture data.
        
        Returns:
            DataFrame with columns: time, participant_id, task, and sensor data
        """
        filepath = self.data_dir / filename
        print(f"Loading XSens data from {filepath}...")
        
        df = pd.read_csv(filepath)
        print(f"  ✓ Loaded {len(df)} frames with {len(df.columns)} columns")
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            print(f"  ⚠ Warning: {missing} missing values found")
            df = df.fillna(method='ffill').fillna(method='bfill')
            print(f"  ✓ Missing values filled")
        
        return df
    
    def load_insoles(self, filename: str = "insoles.csv") -> Optional[pd.DataFrame]:
        """
        Load insole pressure sensor data.
        
        Returns:
            DataFrame with pressure sensor readings, or None if file not found
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"  ⚠ Insole data not found at {filepath}")
            return None
            
        print(f"Loading insole data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"  ✓ Loaded {len(df)} frames")
        
        return df
    
    def load_eyetracker(self, filename: str = "eyetracker.csv") -> Optional[pd.DataFrame]:
        """
        Load eye tracker data.
        
        Returns:
            DataFrame with gaze data, or None if file not found
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"  ⚠ Eye tracker data not found at {filepath}")
            return None
            
        print(f"Loading eye tracker data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"  ✓ Loaded {len(df)} frames")
        
        return df
    
    def load_labels(self, filename: str = "labels.csv") -> Optional[pd.DataFrame]:
        """
        Load ground truth labels if available.
        
        Returns:
            DataFrame with labels, or None if file not found
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"  ⚠ Labels not found at {filepath}")
            return None
            
        print(f"Loading labels from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"  ✓ Loaded {len(df)} labels")
        
        return df
    
    def merge_data(
        self, 
        xsens: pd.DataFrame,
        insoles: Optional[pd.DataFrame] = None,
        eyetracker: Optional[pd.DataFrame] = None,
        labels: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge all data sources on timestamp.
        
        Args:
            xsens: XSens data (required)
            insoles: Insole data (optional)
            eyetracker: Eye tracker data (optional)
            labels: Ground truth labels (optional)
            
        Returns:
            Merged DataFrame with all available data
        """
        print("\nMerging data sources...")
        merged = xsens.copy()
        
        if insoles is not None:
            merged = pd.merge(merged, insoles, on='time', how='left', suffixes=('', '_insole'))
            print(f"  ✓ Merged insole data")
            
        if eyetracker is not None:
            merged = pd.merge(merged, eyetracker, on='time', how='left', suffixes=('', '_eye'))
            print(f"  ✓ Merged eye tracker data")
            
        if labels is not None:
            merged = pd.merge(merged, labels, on='time', how='left')
            print(f"  ✓ Merged labels")
        
        print(f"\nFinal dataset: {len(merged)} frames, {len(merged.columns)} columns")
        return merged
    
    def split_by_participant(
        self, 
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by participant (avoids data leakage).
        
        Args:
            df: Full dataset
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_state: Random seed for reproducibility
            
        Returns:
            train_df, val_df, test_df
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Get unique participants
        if 'participant_id' not in df.columns:
            print("  ⚠ No participant_id column, using temporal split instead")
            return self.split_temporal(df, train_ratio, val_ratio, test_ratio)
        
        participants = df['participant_id'].unique()
        np.random.seed(random_state)
        np.random.shuffle(participants)
        
        n_train = int(len(participants) * train_ratio)
        n_val = int(len(participants) * val_ratio)
        
        train_participants = participants[:n_train]
        val_participants = participants[n_train:n_train + n_val]
        test_participants = participants[n_train + n_val:]
        
        train_df = df[df['participant_id'].isin(train_participants)]
        val_df = df[df['participant_id'].isin(val_participants)]
        test_df = df[df['participant_id'].isin(test_participants)]
        
        print(f"\nData split by participant:")
        print(f"  Train: {len(train_df)} frames ({len(train_participants)} participants)")
        print(f"  Val:   {len(val_df)} frames ({len(val_participants)} participants)")
        print(f"  Test:  {len(test_df)} frames ({len(test_participants)} participants)")
        
        return train_df, val_df, test_df
    
    def split_temporal(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally (earlier data for training).
        
        Args:
            df: Full dataset
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            train_df, val_df, test_df
        """
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]
        
        print(f"\nData split temporally:")
        print(f"  Train: {len(train_df)} frames")
        print(f"  Val:   {len(val_df)} frames")
        print(f"  Test:  {len(test_df)} frames")
        
        return train_df, val_df, test_df


def load_all_data(data_dir: str = "src/data") -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load all available data.
    
    Returns:
        Dictionary with 'xsens', 'insoles', 'eyetracker', 'labels' keys
    """
    loader = DataLoader(data_dir)
    
    data = {
        'xsens': loader.load_xsens(),
        'insoles': loader.load_insoles(),
        'eyetracker': loader.load_eyetracker(),
        'labels': loader.load_labels()
    }
    
    # Remove None values
    data = {k: v for k, v in data.items() if v is not None}
    
    return data


if __name__ == "__main__":
    # Test data loading
    print("=" * 60)
    print("Testing Data Loading")
    print("=" * 60)
    
    loader = DataLoader()
    
    # Load XSens data
    xsens = loader.load_xsens()
    print(f"\nXSens data shape: {xsens.shape}")
    print(f"Time range: {xsens['time'].min():.2f}s to {xsens['time'].max():.2f}s")
    print(f"Columns: {list(xsens.columns[:10])}...")
    
    # Try to load other data
    insoles = loader.load_insoles()
    eyetracker = loader.load_eyetracker()
    labels = loader.load_labels()
    
    # Test data splitting
    train, val, test = loader.split_by_participant(xsens)
    
    print("\n" + "=" * 60)
    print("✓ Data loading successful!")
    print("=" * 60)

