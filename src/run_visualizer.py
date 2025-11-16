import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication

# Import your visualizers
from visualizers.xsens_playback_tool import XsensPlaybackTool
from visualizers.sole_visualizer import SoleVisualizer
from visualizers.eyetracker_visualizer import EyeTrackerVisualizer

# ---- SELECT WHICH VISUALIZER TO RUN ----
RUN_XSENS = True
RUN_INSOLES = True
RUN_EYETRACKER = True

def run_xsens():
    df = pd.read_csv("src/data/xsens.csv")
    app = QApplication(sys.argv)
    w = XsensPlaybackTool(df)
    w.show()
    sys.exit(app.exec_())

def run_insoles():
    df = pd.read_csv("src/data/insoles.csv")
    app = QApplication(sys.argv)
    w = SoleVisualizer(df)
    w.show()
    sys.exit(app.exec_())

def run_eyetracker():
    df = pd.read_csv("src/data/eyetracker.csv")
    video_path = "videos/courseA_2.mp4"   # Using available video
    app = QApplication(sys.argv)
    w = EyeTrackerVisualizer(df, video_path)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    if RUN_XSENS:
        run_xsens()
    if RUN_INSOLES:
        run_insoles()
    if RUN_EYETRACKER:
        run_eyetracker()
