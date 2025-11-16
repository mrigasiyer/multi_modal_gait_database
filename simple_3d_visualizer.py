#!/usr/bin/env python
"""
Simple matplotlib-based 3D visualizer for XSens data
Works on M1/M2/M3 Macs (PyQt5's QGLWidget has rendering issues on Apple Silicon)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QPushButton, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer

class XSens3DVisualizer(QMainWindow):
    def __init__(self, dataframe):
        super().__init__()
        self.df = dataframe
        self.current_frame = 0
        
        self.setWindowTitle("XSens 3D Gait Visualizer (M2 Mac Compatible)")
        self.setGeometry(100, 100, 1000, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        layout.addWidget(self.canvas)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)
        
        # Speed controls
        speed_label = QLabel("Speed:")
        controls_layout.addWidget(speed_label)
        
        self.speed_0_5x = QPushButton("0.5x")
        self.speed_0_5x.clicked.connect(lambda: self.set_speed(0.5))
        controls_layout.addWidget(self.speed_0_5x)
        
        self.speed_1x = QPushButton("1x")
        self.speed_1x.clicked.connect(lambda: self.set_speed(1.0))
        self.speed_1x.setStyleSheet("font-weight: bold;")
        controls_layout.addWidget(self.speed_1x)
        
        self.speed_2x = QPushButton("2x")
        self.speed_2x.clicked.connect(lambda: self.set_speed(2.0))
        controls_layout.addWidget(self.speed_2x)
        
        self.speed_4x = QPushButton("4x")
        self.speed_4x.clicked.connect(lambda: self.set_speed(4.0))
        controls_layout.addWidget(self.speed_4x)
        
        # Frame slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.df) - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slider_change)
        controls_layout.addWidget(self.slider)
        
        # Frame label
        self.frame_label = QLabel(f"Frame: 0 / {len(self.df)-1}")
        controls_layout.addWidget(self.frame_label)
        
        layout.addLayout(controls_layout)
        
        # Animation timer and speed
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.playing = False
        self.playback_speed = 1.0  # 1x speed
        self.frame_skip = 1  # How many frames to skip per update
        
        # Body segment connections (skeleton structure)
        self.connections = [
            # Spine
            ('Pelvis', 'L5'),
            ('L5', 'T8'),
            ('T8', 'Neck'),
            ('Neck', 'Head'),
            # Right arm
            ('T8', 'RightUpperArm'),
            ('RightUpperArm', 'RightForeArm'),
            ('RightForeArm', 'RightHand'),
            # Left arm
            ('T8', 'LeftUpperArm'),
            ('LeftUpperArm', 'LeftForeArm'),
            ('LeftForeArm', 'LeftHand'),
            # Right leg
            ('Pelvis', 'RightUpperLeg'),
            ('RightUpperLeg', 'RightLowerLeg'),
            ('RightLowerLeg', 'RightFoot'),
            ('RightFoot', 'RightToe'),
            # Left leg
            ('Pelvis', 'LeftUpperLeg'),
            ('LeftUpperLeg', 'LeftLowerLeg'),
            ('LeftLowerLeg', 'LeftFoot'),
            ('LeftFoot', 'LeftToe'),
        ]
        
        # Initial plot
        self.update_plot()
        
    def get_segment_position(self, segment_name, row):
        """Get 3D position of a body segment from data row"""
        x = row[f'position_{segment_name}_x']
        y = row[f'position_{segment_name}_y']
        z = row[f'position_{segment_name}_z']
        return np.array([x, y, z])
    
    def update_plot(self):
        """Update the 3D plot with current frame data"""
        self.ax.clear()
        
        # Get current data row
        row = self.df.iloc[self.current_frame]
        
        # Calculate center offset (average of feet positions for stability)
        right_toe_pos = self.get_segment_position('RightToe', row)
        left_toe_pos = self.get_segment_position('LeftToe', row)
        center_offset = (right_toe_pos + left_toe_pos) / 2.0
        center_offset[2] = 0  # Keep z at 0
        
        # Draw skeleton connections
        for seg1, seg2 in self.connections:
            pos1 = self.get_segment_position(seg1, row) - center_offset
            pos2 = self.get_segment_position(seg2, row) - center_offset
            
            # Color code different body parts
            if 'Leg' in seg1 or 'Foot' in seg1 or 'Toe' in seg1:
                color = 'blue' if 'Right' in seg1 else 'green'
            elif 'Arm' in seg1 or 'Hand' in seg1:
                color = 'red' if 'Right' in seg1 else 'orange'
            else:
                color = 'purple'  # Spine
            
            self.ax.plot([pos1[0], pos2[0]], 
                        [pos1[1], pos2[1]], 
                        [pos1[2], pos2[2]], 
                        color=color, linewidth=3, marker='o', markersize=4)
        
        # Draw floor grid
        grid_size = 2
        x_grid = np.linspace(-grid_size, grid_size, 11)
        y_grid = np.linspace(-grid_size, grid_size, 11)
        for x in x_grid:
            self.ax.plot([x, x], [-grid_size, grid_size], [0, 0], 'gray', alpha=0.3, linewidth=0.5)
        for y in y_grid:
            self.ax.plot([-grid_size, grid_size], [y, y], [0, 0], 'gray', alpha=0.3, linewidth=0.5)
        
        # Set labels and limits
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (meters)')
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(0, 2)
        
        # Set view angle
        self.ax.view_init(elev=20, azim=45)
        
        # Update title
        time_sec = row['time'] / 1000.0
        self.ax.set_title(f'Frame {self.current_frame} / {len(self.df)-1} | Time: {time_sec:.2f}s', 
                         fontsize=12, fontweight='bold')
        
        self.canvas.draw()
        self.frame_label.setText(f"Frame: {self.current_frame} / {len(self.df)-1}")
    
    def on_slider_change(self, value):
        """Handle slider value change"""
        self.current_frame = value
        self.update_plot()
    
    def next_frame(self):
        """Advance to next frame"""
        self.current_frame += self.frame_skip
        if self.current_frame >= len(self.df):
            self.current_frame = 0
        self.slider.setValue(self.current_frame)
    
    def set_speed(self, speed):
        """Set playback speed"""
        self.playback_speed = speed
        
        # Update frame skip for faster speeds
        if speed <= 1.0:
            self.frame_skip = 1
            interval = int(100 / speed)  # Slower = longer interval
        else:
            self.frame_skip = int(speed)  # Skip frames for speeds > 1x
            interval = 100
        
        # Update button styles
        for btn in [self.speed_0_5x, self.speed_1x, self.speed_2x, self.speed_4x]:
            btn.setStyleSheet("")
        
        if speed == 0.5:
            self.speed_0_5x.setStyleSheet("font-weight: bold;")
        elif speed == 1.0:
            self.speed_1x.setStyleSheet("font-weight: bold;")
        elif speed == 2.0:
            self.speed_2x.setStyleSheet("font-weight: bold;")
        elif speed == 4.0:
            self.speed_4x.setStyleSheet("font-weight: bold;")
        
        # Restart timer if playing
        if self.playing:
            self.timer.stop()
            self.timer.start(interval)
    
    def toggle_play(self):
        """Toggle play/pause"""
        if self.playing:
            self.timer.stop()
            self.play_button.setText("Play")
            self.playing = False
        else:
            # Calculate interval based on speed
            if self.playback_speed <= 1.0:
                interval = int(100 / self.playback_speed)
            else:
                interval = 100
            self.timer.start(interval)
            self.play_button.setText("Pause")
            self.playing = True

if __name__ == '__main__':
    print("Loading XSens data...")
    df = pd.read_csv("src/data/xsens.csv")
    print(f"✓ Loaded {len(df)} frames")
    
    print("\nCreating 3D visualizer...")
    app = QApplication(sys.argv)
    window = XSens3DVisualizer(df)
    window.show()
    
    print("\n" + "="*60)
    print("MATPLOTLIB 3D VISUALIZER - M2 MAC COMPATIBLE")
    print("="*60)
    print("\nControls:")
    print("- Use the slider to scrub through frames")
    print("- Click 'Play' to animate")
    print("- Speed buttons: 0.5x, 1x, 2x, 4x")
    print("- Rotate view: Click and drag on the 3D plot")
    print("- Zoom: Use scroll wheel on the plot")
    print("\nColor coding:")
    print("  Blue = Right leg")
    print("  Green = Left leg")
    print("  Red = Right arm")
    print("  Orange = Left arm")
    print("  Purple = Spine/Torso")
    print("="*60)
    
    sys.exit(app.exec_())

