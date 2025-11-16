# ML Models for Injury-Prone Position Detection

This directory contains machine learning models for detecting injury-prone positions and safety hazards in gait data.

## 🎯 Goal
Detect biomechanical risk factors that could lead to injury:
- Excessive joint angles
- Poor posture and balance
- Asymmetric gait patterns
- High impact forces
- Unstable foot placement
- Movement degradation (fatigue)

## 📁 Directory Structure

```
ml_models/
├── data/                           # Data preprocessing & feature engineering
│   ├── preprocessing.py            # Load and clean data
│   ├── feature_engineering.py      # Extract biomechanical features
│   └── risk_features.py            # Injury-specific features
├── models/                         # ML models
│   ├── baseline.py                 # Rule-based baseline detector
│   ├── injury_detector.py          # Main ML classifier
│   └── train.py                    # Training pipeline
├── evaluation/                     # Model evaluation
│   ├── metrics.py                  # Performance metrics
│   └── visualize_results.py        # Result visualization
├── notebooks/                      # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── trained_models/                 # Saved model files (gitignored)
├── config/                         # Configuration files
│   └── risk_thresholds.yaml        # Biomechanical thresholds
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd /Users/mrigas/multi_modal_gait_database
source my_venv/bin/activate
pip install -r requirements.txt
```

### 2. Explore Data
```bash
jupyter notebook ml_models/notebooks/01_data_exploration.ipynb
```

### 3. Train Baseline Model
```python
from ml_models.models.baseline import BaselineInjuryDetector

detector = BaselineInjuryDetector()
results = detector.detect('src/data/xsens.csv')
```

### 4. Train ML Model
```bash
python ml_models/models/train.py
```

## 📊 Features Extracted

### Joint Angle Features
- Knee flexion/extension angles
- Hip angles
- Ankle angles
- Spine flexion/rotation
- Joint angle ranges and extremes

### Balance & Stability Features
- Center of mass sway
- Base of support width
- Single vs double support time
- Pressure distribution (from insoles)

### Impact & Force Features
- Peak vertical acceleration
- Loading rates
- Impact asymmetry
- Jerk (smoothness)

### Movement Quality Features
- Step-to-step variability
- Gait symmetry
- Movement smoothness
- Temporal patterns

## 🎓 Approach

1. **Phase 1: Rule-Based Baseline**
   - Define biomechanical thresholds
   - Simple risk scoring
   - Fast, interpretable

2. **Phase 2: Anomaly Detection**
   - Train on normal gait
   - Detect deviations
   - Unsupervised approach

3. **Phase 3: Supervised Classification**
   - Use baseline labels
   - Random Forest / XGBoost
   - Feature importance analysis

4. **Phase 4: Integration**
   - Real-time detection
   - Visualize risks in 3D
   - Production deployment

## 📈 Evaluation Metrics

- Precision: Of flagged risks, how many are real?
- Recall: Of actual risks, how many caught?
- F1-Score: Balance of precision and recall
- False alarm rate: Alarms per minute
- Risk calibration: Score vs actual severity

## 🔗 Integration with Visualizer

Models can be integrated with the 3D visualizer to show real-time risk detection:

```python
from ml_models.models.injury_detector import InjuryDetector

detector = InjuryDetector.load('ml_models/trained_models/model.pkl')
risk_score = detector.predict(current_frame_data)
# Highlight risky joints in red!
```

## 📚 References

Biomechanical risk factors based on gait analysis research and sports medicine guidelines.

