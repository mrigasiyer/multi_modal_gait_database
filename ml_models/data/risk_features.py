"""
Risk feature extraction for injury detection.
Extracts biomechanical features that indicate injury risk.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import yaml
from pathlib import Path


class RiskFeatureExtractor:
    """Extract injury risk features from gait data."""
    
    def __init__(self, config_path: str = "ml_models/config/risk_thresholds.yaml"):
        """
        Initialize feature extractor.
        
        Args:
            config_path: Path to risk thresholds configuration file
        """
        self.config_path = Path(config_path)
        self.thresholds = self._load_thresholds()
        
    def _load_thresholds(self) -> Dict:
        """Load risk thresholds from YAML config."""
        if not self.config_path.exists():
            print(f"⚠ Config file not found: {self.config_path}")
            return {}
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all risk features from data.
        
        Args:
            df: DataFrame with XSens and optionally insole data
            
        Returns:
            DataFrame with extracted features
        """
        print("Extracting risk features...")
        
        features = df[['time']].copy() if 'time' in df.columns else pd.DataFrame(index=df.index)
        
        # Joint angle features
        joint_features = self.extract_joint_angle_features(df)
        features = pd.concat([features, joint_features], axis=1)
        print(f"  ✓ Extracted {len(joint_features.columns)} joint angle features")
        
        # Balance features
        balance_features = self.extract_balance_features(df)
        features = pd.concat([features, balance_features], axis=1)
        print(f"  ✓ Extracted {len(balance_features.columns)} balance features")
        
        # Impact features
        impact_features = self.extract_impact_features(df)
        features = pd.concat([features, impact_features], axis=1)
        print(f"  ✓ Extracted {len(impact_features.columns)} impact features")
        
        # Movement quality features
        quality_features = self.extract_movement_quality_features(df)
        features = pd.concat([features, quality_features], axis=1)
        print(f"  ✓ Extracted {len(quality_features.columns)} movement quality features")
        
        # Asymmetry features
        asymmetry_features = self.extract_asymmetry_features(df)
        features = pd.concat([features, asymmetry_features], axis=1)
        print(f"  ✓ Extracted {len(asymmetry_features.columns)} asymmetry features")
        
        print(f"\nTotal features extracted: {len(features.columns)}")
        return features
    
    def extract_joint_angle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract joint angle based features."""
        features = pd.DataFrame(index=df.index)
        
        # Key joints to analyze
        joints = ['jRightKnee', 'jLeftKnee', 'jRightAnkle', 'jLeftAnkle', 
                  'jRightHip', 'jLeftHip']
        
        for joint in joints:
            # Check if joint angle columns exist
            col_x = f'jointAngle_{joint}_x'
            col_y = f'jointAngle_{joint}_y'
            col_z = f'jointAngle_{joint}_z'
            
            if col_x in df.columns:
                # Extract angles (convert to degrees if in radians)
                angle_x = np.degrees(df[col_x]) if df[col_x].abs().max() < 10 else df[col_x]
                angle_y = np.degrees(df[col_y]) if df[col_y].abs().max() < 10 else df[col_y]
                angle_z = np.degrees(df[col_z]) if df[col_z].abs().max() < 10 else df[col_z]
                
                # Primary angle (flexion/extension - typically y or z axis)
                features[f'{joint}_angle'] = angle_y
                
                # Magnitude (total deviation from neutral)
                features[f'{joint}_magnitude'] = np.sqrt(angle_x**2 + angle_y**2 + angle_z**2)
                
                # Extreme angles (risk indicators)
                features[f'{joint}_extreme'] = (features[f'{joint}_angle'].abs() > 90).astype(int)
        
        return features
    
    def extract_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract balance and stability features."""
        features = pd.DataFrame(index=df.index)
        
        # Center of Mass (approximate using Pelvis position)
        if 'position_Pelvis_x' in df.columns:
            com_x = df['position_Pelvis_x']
            com_y = df['position_Pelvis_y']
            com_z = df['position_Pelvis_z']
            
            # Lateral sway (side-to-side movement)
            window = 30  # ~0.5 seconds at 60Hz
            features['com_lateral_sway'] = com_x.rolling(window, center=True).std() * 100  # cm
            features['com_anteroposterior_sway'] = com_y.rolling(window, center=True).std() * 100
            features['com_vertical_oscillation'] = com_z.rolling(window, center=True).std() * 100
            
            # Velocity of CoM
            features['com_velocity_x'] = com_x.diff().abs()
            features['com_velocity_y'] = com_y.diff().abs()
            features['com_velocity_magnitude'] = np.sqrt(
                features['com_velocity_x']**2 + features['com_velocity_y']**2
            )
        
        # Base of support (step width)
        if 'position_RightFoot_x' in df.columns and 'position_LeftFoot_x' in df.columns:
            right_foot_x = df['position_RightFoot_x']
            left_foot_x = df['position_LeftFoot_x']
            
            features['step_width'] = (right_foot_x - left_foot_x).abs() * 100  # cm
            
            # Detect narrow/wide steps (risk indicators)
            features['step_too_narrow'] = (features['step_width'] < 5).astype(int)
            features['step_too_wide'] = (features['step_width'] > 20).astype(int)
        
        # Trunk lean (from T8 or L5)
        if 'position_T8_z' in df.columns and 'position_Pelvis_z' in df.columns:
            trunk_lean_y = df['position_T8_y'] - df['position_Pelvis_y']
            trunk_height = df['position_T8_z'] - df['position_Pelvis_z']
            
            # Lean angle (forward/backward)
            features['trunk_lean_angle'] = np.degrees(np.arctan2(trunk_lean_y, trunk_height))
            features['excessive_lean'] = (features['trunk_lean_angle'].abs() > 30).astype(int)
        
        return features
    
    def extract_impact_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract impact and force features."""
        features = pd.DataFrame(index=df.index)
        
        # Vertical acceleration (impact indicator)
        if 'acceleration_Pelvis_z' in df.columns:
            accel_z = df['acceleration_Pelvis_z']
            
            # Peak acceleration
            features['peak_vertical_accel'] = accel_z.abs()
            features['high_impact'] = (features['peak_vertical_accel'] > 20).astype(int)  # >2g
            
            # Jerk (rate of acceleration change - smoothness indicator)
            features['vertical_jerk'] = accel_z.diff().abs()
            features['high_jerk'] = (features['vertical_jerk'] > 40).astype(int)
        
        # Foot impact (from foot acceleration)
        for foot in ['RightFoot', 'LeftFoot']:
            accel_col = f'acceleration_{foot}_z'
            if accel_col in df.columns:
                features[f'{foot}_impact'] = df[accel_col].abs()
                
                # Detect heel strikes (large negative acceleration)
                features[f'{foot}_heel_strike'] = (df[accel_col] < -15).astype(int)
        
        return features
    
    def extract_movement_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract movement quality and smoothness features."""
        features = pd.DataFrame(index=df.index)
        
        # Step detection (simplified - based on foot height)
        if 'position_RightFoot_z' in df.columns:
            right_foot_z = df['position_RightFoot_z']
            left_foot_z = df['position_LeftFoot_z']
            
            # Toe clearance (minimum height during swing)
            window = 30
            features['right_toe_clearance'] = right_foot_z.rolling(window).min() * 100  # cm
            features['left_toe_clearance'] = left_foot_z.rolling(window).min() * 100
            
            # Low toe clearance = tripping risk
            features['right_trip_risk'] = (features['right_toe_clearance'] < 2).astype(int)
            features['left_trip_risk'] = (features['left_toe_clearance'] < 2).astype(int)
            
            # Foot dragging detection
            features['foot_dragging'] = ((features['right_toe_clearance'] < 1) | 
                                         (features['left_toe_clearance'] < 1)).astype(int)
        
        # Movement smoothness (from velocity changes)
        if 'velocity_Pelvis_x' in df.columns:
            vel_x = df['velocity_Pelvis_x']
            vel_y = df['velocity_Pelvis_y']
            
            # Speed variability
            window = 60
            features['speed_variability'] = vel_x.rolling(window).std()
            
            # Sudden changes in velocity
            features['sudden_deceleration'] = (vel_x.diff() < -0.5).astype(int)
            features['sudden_acceleration'] = (vel_x.diff() > 0.5).astype(int)
        
        return features
    
    def extract_asymmetry_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract left-right asymmetry features."""
        features = pd.DataFrame(index=df.index)
        
        # Joint angle asymmetry
        joint_pairs = [
            ('jRightKnee', 'jLeftKnee'),
            ('jRightHip', 'jLeftHip'),
            ('jRightAnkle', 'jLeftAnkle')
        ]
        
        for right_joint, left_joint in joint_pairs:
            right_col = f'jointAngle_{right_joint}_y'
            left_col = f'jointAngle_{left_joint}_y'
            
            if right_col in df.columns and left_col in df.columns:
                right_angle = df[right_col]
                left_angle = df[left_col]
                
                # Absolute difference
                features[f'{right_joint}_asymmetry'] = (right_angle - left_angle).abs()
                
                # Percent difference
                mean_angle = (right_angle.abs() + left_angle.abs()) / 2
                features[f'{right_joint}_asymmetry_pct'] = (
                    features[f'{right_joint}_asymmetry'] / (mean_angle + 0.01)
                )
                
                # High asymmetry flag
                features[f'{right_joint}_high_asymmetry'] = (
                    features[f'{right_joint}_asymmetry_pct'] > 0.20
                ).astype(int)
        
        # Position asymmetry (step length, stance time, etc.)
        if 'position_RightFoot_y' in df.columns and 'position_LeftFoot_y' in df.columns:
            right_foot_y = df['position_RightFoot_y']
            left_foot_y = df['position_LeftFoot_y']
            
            # Stride length asymmetry (simplified)
            window = 60
            right_stride = right_foot_y.rolling(window).max() - right_foot_y.rolling(window).min()
            left_stride = left_foot_y.rolling(window).max() - left_foot_y.rolling(window).min()
            
            features['stride_asymmetry'] = (right_stride - left_stride).abs()
            features['stride_asymmetry_pct'] = features['stride_asymmetry'] / ((right_stride + left_stride) / 2 + 0.01)
        
        return features
    
    def compute_risk_score(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Compute overall risk score from features.
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            DataFrame with risk scores added
        """
        if not self.thresholds:
            print("⚠ No thresholds loaded, cannot compute risk scores")
            return features
        
        risk_scores = pd.DataFrame(index=features.index)
        
        # Joint angle risk
        if 'jRightKnee_angle' in features.columns:
            knee_risk = features['jRightKnee_extreme'] | features.get('jLeftKnee_extreme', 0)
            risk_scores['joint_angle_risk'] = knee_risk
        
        # Balance risk
        if 'com_lateral_sway' in features.columns:
            balance_risk = ((features['com_lateral_sway'] > 5) | 
                           (features.get('excessive_lean', 0) > 0))
            risk_scores['balance_risk'] = balance_risk.astype(int)
        
        # Impact risk
        if 'high_impact' in features.columns:
            risk_scores['impact_risk'] = features['high_impact']
        
        # Movement quality risk
        if 'foot_dragging' in features.columns:
            quality_risk = features['foot_dragging'] | features.get('right_trip_risk', 0) | features.get('left_trip_risk', 0)
            risk_scores['movement_quality_risk'] = quality_risk.astype(int)
        
        # Asymmetry risk
        asymmetry_cols = [c for c in features.columns if 'high_asymmetry' in c]
        if asymmetry_cols:
            risk_scores['asymmetry_risk'] = features[asymmetry_cols].any(axis=1).astype(int)
        
        # Overall risk score (weighted sum)
        weights = self.thresholds.get('risk_weights', {})
        risk_scores['overall_risk_score'] = (
            risk_scores.get('joint_angle_risk', 0) * weights.get('joint_angles', 0.25) +
            risk_scores.get('balance_risk', 0) * weights.get('balance', 0.25) +
            risk_scores.get('impact_risk', 0) * weights.get('impact', 0.20) +
            risk_scores.get('movement_quality_risk', 0) * weights.get('movement_quality', 0.10) +
            risk_scores.get('asymmetry_risk', 0) * weights.get('temporal', 0.05)
        )
        
        # Risk level classification
        risk_levels = self.thresholds.get('risk_levels', {})
        risk_scores['risk_level'] = pd.cut(
            risk_scores['overall_risk_score'],
            bins=[0, risk_levels['low'][1], risk_levels['medium'][1], 1.0],
            labels=['low', 'medium', 'high']
        )
        
        # Add risk scores to features
        result = pd.concat([features, risk_scores], axis=1)
        
        return result


if __name__ == "__main__":
    # Test feature extraction
    from preprocessing import DataLoader
    
    print("=" * 60)
    print("Testing Feature Extraction")
    print("=" * 60)
    
    # Load data
    loader = DataLoader()
    df = loader.load_xsens()
    
    # Extract features
    extractor = RiskFeatureExtractor()
    features = extractor.extract_all_features(df.iloc[:1000])  # Test on first 1000 frames
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"\nSample features:")
    print(features.head())
    
    # Compute risk scores
    features_with_risk = extractor.compute_risk_score(features)
    
    print(f"\nRisk scores computed:")
    print(features_with_risk[['overall_risk_score', 'risk_level']].describe())
    
    print("\n" + "=" * 60)
    print("✓ Feature extraction successful!")
    print("=" * 60)

