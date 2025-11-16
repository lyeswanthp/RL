#!/usr/bin/env python3
"""
Test to verify that stay_id is preserved during data cleaning
"""

import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.data_cleaning import DataCleaner
from utils.config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_stay_id_preservation():
    """Test that stay_id is preserved during cleaning"""
    print("\n" + "="*80)
    print("Testing stay_id preservation during data cleaning")
    print("="*80)

    # Load config
    config = ConfigLoader().config
    cleaner = DataCleaner(config)

    # Create sample data with stay_id and time_window
    np.random.seed(42)
    n_stays = 100
    n_windows_per_stay = 10
    total_rows = n_stays * n_windows_per_stay

    sample_data = pd.DataFrame({
        'stay_id': np.repeat(range(n_stays), n_windows_per_stay),
        'time_window': np.tile(range(n_windows_per_stay), n_stays),
        'HR': np.random.normal(80, 15, total_rows),
        'SpO2': np.random.normal(97, 2, total_rows),
        'RR': np.random.normal(16, 4, total_rows),
        'age': np.repeat(np.random.randint(18, 90, n_stays), n_windows_per_stay),
        'gender': np.repeat(np.random.choice([0, 1], n_stays), n_windows_per_stay),
    })

    # Add some missing values
    sample_data.loc[15:20, 'HR'] = np.nan
    sample_data.loc[30:35, 'SpO2'] = np.nan
    sample_data.loc[50:55, 'RR'] = np.nan

    print(f"\nBefore cleaning:")
    print(f"  Shape: {sample_data.shape}")
    print(f"  Columns: {list(sample_data.columns)}")
    print(f"  Has stay_id: {'stay_id' in sample_data.columns}")
    print(f"  Has time_window: {'time_window' in sample_data.columns}")
    print(f"  Missing values: {sample_data.isnull().sum().sum()}")

    # Clean data
    cleaned_data, stats = cleaner.clean_data(sample_data, temporal=True)

    print(f"\nAfter cleaning:")
    print(f"  Shape: {cleaned_data.shape}")
    print(f"  Columns: {list(cleaned_data.columns)}")
    print(f"  Has stay_id: {'stay_id' in cleaned_data.columns}")
    print(f"  Has time_window: {'time_window' in cleaned_data.columns}")
    print(f"  Missing values: {cleaned_data.isnull().sum().sum()}")

    # Verify stay_id is preserved
    if 'stay_id' not in cleaned_data.columns:
        print("\n❌ FAILED: stay_id column is missing!")
        return False

    if 'time_window' not in cleaned_data.columns:
        print("\n❌ FAILED: time_window column is missing!")
        return False

    # Verify all stay_ids are present
    original_stays = set(sample_data['stay_id'].unique())
    cleaned_stays = set(cleaned_data['stay_id'].unique())

    if original_stays != cleaned_stays:
        print(f"\n❌ FAILED: Some stay_ids were lost!")
        print(f"  Original: {len(original_stays)} stays")
        print(f"  Cleaned: {len(cleaned_stays)} stays")
        return False

    # Verify no missing values remain
    if cleaned_data.isnull().sum().sum() > 0:
        print(f"\n⚠️  WARNING: {cleaned_data.isnull().sum().sum()} missing values remain")

    print("\n✅ SUCCESS: stay_id and time_window are properly preserved!")
    print("\nCleaning stats:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    return True

if __name__ == "__main__":
    success = test_stay_id_preservation()
    sys.exit(0 if success else 1)
