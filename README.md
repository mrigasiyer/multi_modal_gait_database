# Multi Modal Gait Database

This is the repository to the article<br/>
V.Losing & M. Hasenjäger, _A Multi-Modal Gait Database of Natural Everyday-Walk in an Urban Environment_, 2022. We provide some scripts to generate custom pandas data frames from the raw data to streamline the processing and machine learning. Additionally, there is also a visualization / labeling tool enabling data inspection and the modification of the labels.

## Example videos

These videos provide an impression of the available data that is based on recordings using the XSens motion suit, insoles pressure sensors as well as an eye tracker.

- [course A](https://github.com/HRI-EU/multi_modal_gait_database/blob/master/videos/courseA_2.mp4)
- [course B](https://github.com/HRI-EU/multi_modal_gait_database/blob/master/videos/courseB_2.mp4)
- [course C](https://github.com/HRI-EU/multi_modal_gait_database/blob/master/videos/courseC_2.mp4)

## Quick Start: Running the 3D Visualizer

### Prerequisites

- Python 3.9 or higher
- macOS, Linux, or Windows
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/multi_modal_gait_database.git
cd multi_modal_gait_database
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv my_venv
source my_venv/bin/activate  # On Windows: my_venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add Your Data Files

The repository excludes large CSV data files. You need to add your own data files to `src/data/`:

```bash
# Create the data directory if it doesn't exist
mkdir -p src/data

# Add your CSV files (download these from the dataset)
# Place the following files in src/data/:
#   - xsens.csv      (XSens motion capture data)
#   - insoles.csv    (Pressure sensor data)
#   - eyetracker.csv (Eye tracking data)
#   - labels.csv     (Ground truth labels)
```

### Step 5: Run the 3D Visualizer

```bash
python simple_3d_visualizer.py
```

This will open an interactive 3D window showing the human gait animation!

**Controls:**

- **Play/Pause button**: Start/stop animation
- **Speed buttons**: 0.5x, 1x, 2x, 4x playback speed
- **Slider**: Scrub through frames manually
- **Mouse drag**: Rotate the 3D view
- **Scroll wheel**: Zoom in/out

**Note for M1/M2/M3 Mac users:** The `simple_3d_visualizer.py` uses matplotlib for rendering, which works perfectly on Apple Silicon. The original OpenGL-based visualizer has known compatibility issues with PyQt5's deprecated QGLWidget on M-series Macs.

### Deactivating Virtual Environment

When you're done:

```bash
deactivate
```

## Machine Learning: Injury Detection

This repository includes machine learning models for detecting injury-prone positions and safety hazards from gait data.

### Quick Start: Run Baseline Injury Detector

```bash
# Activate virtual environment
source my_venv/bin/activate

# Test the ML setup
python ml_models/test_setup.py

# Run baseline injury detector on your data
python ml_models/models/baseline.py --input src/data/xsens.csv
```

This will analyze your gait data and identify frames with potential injury risk based on biomechanical thresholds.

### What Gets Detected?

The injury detector identifies:

- **Excessive joint angles** (knee, hip, ankle hyperflexion)
- **Poor balance** (excessive center of mass sway, unstable stance)
- **High impact forces** (excessive vertical acceleration, jerky movements)
- **Movement degradation** (low toe clearance = tripping risk, foot dragging)
- **Asymmetric patterns** (left-right differences indicating compensation or limping)

### ML Model Structure

```
ml_models/
├── data/                      # Data preprocessing & feature engineering
│   ├── preprocessing.py       # Load and prepare data
│   └── risk_features.py       # Extract biomechanical features
├── models/                    # ML models
│   ├── baseline.py            # Rule-based detector (ready to use)
│   └── train.py               # Advanced ML training (coming soon)
├── notebooks/                 # Jupyter notebooks for exploration
│   └── 01_data_exploration.ipynb
├── config/                    # Configuration files
│   └── risk_thresholds.yaml   # Biomechanical safety thresholds
└── README.md                  # Detailed ML documentation
```

### Explore Data Interactively

Launch Jupyter to explore the data and features:

```bash
jupyter notebook ml_models/notebooks/01_data_exploration.ipynb
```

### Advanced Usage

**Custom Thresholds:**

Edit `ml_models/config/risk_thresholds.yaml` to adjust safety thresholds based on your needs.

**Export Annotations:**

```bash
python ml_models/models/baseline.py \
  --input src/data/xsens.csv \
  --output results.csv \
  --annotations hazard_segments.csv
```

**Integration with Visualizer:**

The ML models can be integrated with the 3D visualizer to show real-time risk detection (coming soon!).

### Next Steps

1. **Feature Engineering**: Extract custom biomechanical features
2. **Train ML Models**: Random Forest, XGBoost for improved detection
3. **Real-time Detection**: Integrate with live visualization
4. **Validation**: Test on unseen participants

See `ml_models/README.md` for detailed documentation.

## Full Setup for Data Processing

## Generate pandas data frame from .csv files

We provide a script that generates one pandas data frame stored as pickle file from the single recording .csv files. This is quite handy for further processing or analysis.<br/>
`PYTHONPATH=$(pwd) python src/data_frame_extractor.py -s 1,2 -c a,b -p xxxx/NEWBEE_dataset/data_set/ -d destination_file_path`

- -s 1,2 for subjects 1 and 2 (default is all subjects)
- -c a,b for course A and B (default is all courses A,B,C)
- -p path to the data set
- -d path to the data frame stored as pickle file
  (more parameters are available see `python src/data_frame_extractor.py --help`)

## Labeling tool

The tool allows to inspect the data but also to change the labels or use even custom labels.

It can be started by:<br/>
`PYTHONPATH=$(pwd) python src/labeling_tool.py -s 1 -c a -p xxxx/NEWBEE_dataset/data_set/`<br/>

- -s 1 for subject 1
- -c a for course A
- -p path to the data set

## Insole pressure danger detection

The repository also contains a small utility that inspects the normalized insole
pressure channels and highlights frames that might correspond to hazardous
stances (e.g. extreme forefoot loading or sudden pressure shifts).

### Running the analysis

1. Ensure your virtual environment is activated and dependencies are installed
   (see the [Setup](#setup) section above).
2. Invoke the CLI with the CSV that came from `data_frame_extractor.py` or any
   file that follows the same column naming scheme:

```shell
PYTHONPATH=$(pwd) python src/insole_danger_detection.py path/to/insole_only.csv \
  --output annotated.csv
```

The script prints a short summary of the triggered danger conditions. Supplying
`--output` is optional; if present, the annotated CSV (including engineered
features and boolean danger flags) is written to that location.

### Adjusting thresholds

All heuristics are configurable via CLI switches. For example, to raise the
single-sensor overload threshold and require a larger jump in total pressure
before flagging sudden changes:

```shell
PYTHONPATH=$(pwd) python src/insole_danger_detection.py path/to/insole_only.csv \
  --max-pressure 0.95 --pressure-jump 0.35
```

Run `PYTHONPATH=$(pwd) python src/insole_danger_detection.py --help` to see the
full list of tunable parameters.

### Quick smoke test

Because the module is pure Python, you can perform a basic syntax check with
`python -m compileall`:

```shell
python -m compileall src/insole_danger_detection.py
```

This command completes silently if the file is syntactically correct.

## Available Visualizers

### 1. Simple 3D Visualizer (Recommended for M1/M2/M3 Macs)

**File:** `simple_3d_visualizer.py`

A matplotlib-based 3D visualizer that works on all platforms, especially Apple Silicon Macs.

**Features:**

- Interactive 3D skeleton visualization
- Playback controls with variable speed (0.5x, 1x, 2x, 4x)
- Timeline scrubber
- Mouse-controlled rotation and zoom
- Color-coded body segments

**Run it:**

```bash
python simple_3d_visualizer.py
```

### 2. Original OpenGL Visualizer (Legacy)

**File:** `src/run_visualizer.py`

The original OpenGL-based visualizer with more detailed 3D models.

**Known Issues:**

- Does not render properly on M1/M2/M3 Macs due to PyQt5's QGLWidget compatibility issues
- Works on Intel Macs and Linux systems

**Run it:**

```bash
PYTHONPATH=$(pwd) python src/run_visualizer.py
```

## Troubleshooting

### Virtual Environment Issues

If you have trouble activating the virtual environment:

```bash
# Make sure you're in the project directory
cd multi_modal_gait_database

# Try recreating the virtual environment
rm -rf my_venv
python3 -m venv my_venv
source my_venv/bin/activate
pip install -r requirements.txt
```

### Missing Data Files

If you see "File not found" errors for CSV files:

1. Make sure the `src/data/` directory exists
2. Download the dataset from the original source
3. Place the CSV files in `src/data/` directory

### Black Screen on M1/M2/M3 Mac

If the OpenGL visualizer shows only a black screen:

- This is a known issue with PyQt5's QGLWidget on Apple Silicon
- **Solution:** Use `simple_3d_visualizer.py` instead

### Import Errors

If you see module import errors:

```bash
# Make sure virtual environment is activated
source my_venv/bin/activate  # Should show (my_venv) in your prompt

# Reinstall requirements
pip install --upgrade -r requirements.txt
```

### Python Version Issues

This project requires Python 3.9 or higher. Check your version:

```bash
python3 --version
```

If you need to use a specific Python version:

```bash
python3.9 -m venv my_venv  # Replace 3.9 with your version
```

## Project Structure

```
multi_modal_gait_database/
├── simple_3d_visualizer.py       # M2 Mac compatible visualizer
├── requirements.txt              # Python dependencies
├── src/
│   ├── data/                     # CSV data files (not in repo)
│   │   ├── xsens.csv
│   │   ├── insoles.csv
│   │   ├── eyetracker.csv
│   │   └── labels.csv
│   ├── visualizers/              # Visualization modules
│   ├── run_visualizer.py         # Legacy OpenGL visualizer
│   ├── data_frame_extractor.py   # Data processing
│   ├── labeling_tool.py          # Labeling interface
│   └── insole_danger_detection.py # Safety analysis
└── videos/                       # Example videos
    ├── courseA_2.mp4
    ├── courseB_2.mp4
    └── courseC_2.mp4
```

## Citation

If you use this dataset, please cite:

```
V. Losing & M. Hasenjäger, "A Multi-Modal Gait Database of Natural Everyday-Walk
in an Urban Environment", 2022.
```
