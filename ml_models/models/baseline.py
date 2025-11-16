"""
Baseline rule-based injury detector.
Uses biomechanical thresholds to detect injury-prone positions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_models.data.preprocessing import DataLoader
from ml_models.data.risk_features import RiskFeatureExtractor


class BaselineInjuryDetector:
    """
    Rule-based baseline for injury detection.
    Uses biomechanical thresholds to flag risky positions.
    """
    
    def __init__(self, config_path: str = "ml_models/config/risk_thresholds.yaml"):
        """
        Initialize baseline detector.
        
        Args:
            config_path: Path to risk thresholds configuration
        """
        self.feature_extractor = RiskFeatureExtractor(config_path)
        self.thresholds = self.feature_extractor.thresholds
        
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect injury-prone positions in gait data.
        
        Args:
            data: DataFrame with XSens data
            
        Returns:
            DataFrame with risk scores and flags
        """
        print("Running baseline injury detection...")
        
        # Extract features
        features = self.feature_extractor.extract_all_features(data)
        
        # Compute risk scores
        results = self.feature_extractor.compute_risk_score(features)
        
        # Add binary hazard flag
        results['is_hazard'] = (results['overall_risk_score'] > 0.6).astype(int)
        
        return results
    
    def detect_from_file(self, filepath: str) -> pd.DataFrame:
        """
        Detect injury-prone positions from CSV file.
        
        Args:
            filepath: Path to XSens CSV file
            
        Returns:
            DataFrame with risk scores and flags
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        return self.detect(df)
    
    def summarize_results(self, results: pd.DataFrame) -> dict:
        """
        Summarize detection results.
        
        Args:
            results: DataFrame with detection results
            
        Returns:
            Dictionary with summary statistics
        """
        total_frames = len(results)
        hazard_frames = results['is_hazard'].sum()
        hazard_rate = hazard_frames / total_frames * 100
        
        summary = {
            'total_frames': total_frames,
            'hazard_frames': hazard_frames,
            'safe_frames': total_frames - hazard_frames,
            'hazard_rate_percent': hazard_rate,
            'risk_score_mean': results['overall_risk_score'].mean(),
            'risk_score_std': results['overall_risk_score'].std(),
            'risk_score_max': results['overall_risk_score'].max(),
        }
        
        # Risk level distribution
        if 'risk_level' in results.columns:
            risk_counts = results['risk_level'].value_counts()
            summary['low_risk_frames'] = risk_counts.get('low', 0)
            summary['medium_risk_frames'] = risk_counts.get('medium', 0)
            summary['high_risk_frames'] = risk_counts.get('high', 0)
        
        # Specific risk factors
        risk_columns = [c for c in results.columns if '_risk' in c and c != 'overall_risk_score']
        for col in risk_columns:
            if col in results.columns:
                summary[f'{col}_count'] = results[col].sum()
        
        return summary
    
    def print_summary(self, results: pd.DataFrame):
        """
        Print formatted summary of detection results.
        
        Args:
            results: DataFrame with detection results
        """
        summary = self.summarize_results(results)
        
        print("\n" + "=" * 60)
        print("INJURY DETECTION SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal frames analyzed: {summary['total_frames']:,}")
        print(f"Safe frames:           {summary['safe_frames']:,} ({100 - summary['hazard_rate_percent']:.1f}%)")
        print(f"Hazard frames:         {summary['hazard_frames']:,} ({summary['hazard_rate_percent']:.1f}%)")
        
        print(f"\nRisk Score Statistics:")
        print(f"  Mean:  {summary['risk_score_mean']:.3f}")
        print(f"  Std:   {summary['risk_score_std']:.3f}")
        print(f"  Max:   {summary['risk_score_max']:.3f}")
        
        if 'low_risk_frames' in summary:
            print(f"\nRisk Level Distribution:")
            print(f"  Low:    {summary['low_risk_frames']:,} frames")
            print(f"  Medium: {summary['medium_risk_frames']:,} frames")
            print(f"  High:   {summary['high_risk_frames']:,} frames")
        
        print(f"\nSpecific Risk Factors Detected:")
        risk_factors = [k for k in summary.keys() if '_risk_count' in k]
        for factor in risk_factors:
            factor_name = factor.replace('_risk_count', '').replace('_', ' ').title()
            count = summary[factor]
            if count > 0:
                pct = count / summary['total_frames'] * 100
                print(f"  {factor_name}: {count:,} frames ({pct:.1f}%)")
        
        print("\n" + "=" * 60)
    
    def get_hazard_segments(self, results: pd.DataFrame, min_duration: int = 5) -> list:
        """
        Extract continuous hazard segments.
        
        Args:
            results: DataFrame with detection results
            min_duration: Minimum frames for a segment to be considered
            
        Returns:
            List of (start_frame, end_frame, risk_score) tuples
        """
        hazards = results['is_hazard'].values
        segments = []
        
        in_hazard = False
        start_frame = 0
        
        for i, is_hazard in enumerate(hazards):
            if is_hazard and not in_hazard:
                # Start of hazard segment
                start_frame = i
                in_hazard = True
            elif not is_hazard and in_hazard:
                # End of hazard segment
                if i - start_frame >= min_duration:
                    segment_risk = results.iloc[start_frame:i]['overall_risk_score'].mean()
                    segments.append((start_frame, i, segment_risk))
                in_hazard = False
        
        # Handle case where hazard extends to end
        if in_hazard and len(hazards) - start_frame >= min_duration:
            segment_risk = results.iloc[start_frame:]['overall_risk_score'].mean()
            segments.append((start_frame, len(hazards), segment_risk))
        
        return segments
    
    def export_hazard_annotations(self, results: pd.DataFrame, output_path: str):
        """
        Export hazard detections to CSV for visualization.
        
        Args:
            results: DataFrame with detection results
            output_path: Path to save annotations
        """
        segments = self.get_hazard_segments(results)
        
        annotations = []
        for start, end, risk in segments:
            annotations.append({
                'start_frame': start,
                'end_frame': end,
                'duration_frames': end - start,
                'avg_risk_score': risk,
                'start_time': results.iloc[start]['time'] if 'time' in results.columns else start,
                'end_time': results.iloc[end-1]['time'] if 'time' in results.columns else end,
            })
        
        df_annotations = pd.DataFrame(annotations)
        df_annotations.to_csv(output_path, index=False)
        
        print(f"\n✓ Exported {len(annotations)} hazard segments to {output_path}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Baseline injury detection')
    parser.add_argument('--input', type=str, default='src/data/xsens.csv',
                      help='Input XSens CSV file')
    parser.add_argument('--output', type=str, default='ml_models/results_baseline.csv',
                      help='Output file for results')
    parser.add_argument('--annotations', type=str, default='ml_models/hazard_annotations.csv',
                      help='Output file for hazard annotations')
    
    args = parser.parse_args()
    
    # Run detection
    detector = BaselineInjuryDetector()
    results = detector.detect_from_file(args.input)
    
    # Print summary
    detector.print_summary(results)
    
    # Save results
    results.to_csv(args.output, index=False)
    print(f"\n✓ Full results saved to {args.output}")
    
    # Export annotations
    detector.export_hazard_annotations(results, args.annotations)


if __name__ == "__main__":
    main()

