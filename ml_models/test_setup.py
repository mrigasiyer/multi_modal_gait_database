"""
Test script to verify ML setup is working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 60)
print("Testing ML Model Setup")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from ml_models.data.preprocessing import DataLoader
    from ml_models.data.risk_features import RiskFeatureExtractor
    from ml_models.models.baseline import BaselineInjuryDetector
    import pandas as pd
    import numpy as np
    import sklearn
    import xgboost
    print("  ✓ All imports successful")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Test data loading
print("\n2. Testing data loading...")
try:
    loader = DataLoader(data_dir="src/data")
    df = loader.load_xsens()
    print(f"  ✓ Loaded {len(df)} frames")
    print(f"  ✓ Time range: {df['time'].min():.2f}s to {df['time'].max():.2f}s")
except Exception as e:
    print(f"  ✗ Data loading error: {e}")
    sys.exit(1)

# Test feature extraction
print("\n3. Testing feature extraction...")
try:
    extractor = RiskFeatureExtractor()
    
    # Test on small subset
    test_df = df.iloc[:500]  # First 500 frames
    features = extractor.extract_all_features(test_df)
    
    print(f"  ✓ Extracted {features.shape[1]} features from {features.shape[0]} frames")
    print(f"  ✓ Features include: {list(features.columns[:5])}...")
except Exception as e:
    print(f"  ✗ Feature extraction error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test baseline detector
print("\n4. Testing baseline injury detector...")
try:
    detector = BaselineInjuryDetector()
    
    # Test on small subset
    test_df = df.iloc[:500]
    results = detector.detect(test_df)
    
    summary = detector.summarize_results(results)
    
    print(f"  ✓ Processed {summary['total_frames']} frames")
    print(f"  ✓ Detected {summary['hazard_frames']} hazard frames ({summary['hazard_rate_percent']:.1f}%)")
    print(f"  ✓ Mean risk score: {summary['risk_score_mean']:.3f}")
except Exception as e:
    print(f"  ✗ Detector error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "=" * 60)
print("✓ All tests passed! ML setup is working correctly.")
print("=" * 60)
print("\nNext steps:")
print("  1. Run full detection: python ml_models/models/baseline.py")
print("  2. Explore data: jupyter notebook ml_models/notebooks/01_data_exploration.ipynb")
print("  3. Train ML models: python ml_models/models/train.py (coming soon)")
print("=" * 60)

