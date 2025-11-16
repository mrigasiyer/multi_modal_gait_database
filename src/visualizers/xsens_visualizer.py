# Copyright (C)
# Honda Research Institute Europe GmbH
# Carl-Legien-Str. 30
# 63073 Offenbach/Main
# Germany
#
# UNPUBLISHED PROPRIETARY MATERIAL.
# ALL RIGHTS RESERVED.
__author__ = "Viktor Losing"
__maintainer__ = "Viktor Losing"
__email__ = "viktor.losing@honda-ri.de"

import itertools

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from .lcx.openGLWidget import openGLWidget


class XSensVisualizer(QWidget):
    WINDOW_WIDTH = 500
    WINDOW_HEIGHT = 500
    VISUALIZATION_FRAME_RATE = 10

    def __init__(self):
        super().__init__(parent=None)
        self.xyz = ['_x', '_y', '_z']
        self.quaternions = ['_q1', '_qi', '_qj', '_qk']
        #self.xyz = ['_0', '_1', '_2']
        #self.quaternions = ['_0', '_1', '_2', '_3']
        self.skeleton_data = {
            'Pelvis': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},
            'L5': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},
            'T8': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},
            'Neck': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},
            'Head': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},

            'RightUpperLeg': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},
            'RightLowerLeg': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},
            'RightFoot':     {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},
            'RightToe':      {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},

            'LeftUpperLeg': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},
            'LeftLowerLeg': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},
            'LeftFoot':     {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},
            'LeftToe':      {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},

            'RightUpperArm': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},
            'RightForeArm':  {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},
            'RightHand':     {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},

            'LeftUpperArm': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},
            'LeftForeArm':  {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]},
            'LeftHand':     {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 0]}
        }

        self.resize(XSensVisualizer.WINDOW_WIDTH, XSensVisualizer.WINDOW_HEIGHT)

        self.mapped_segment_names = [
            'Pelvis','L5','T8','Neck','Head',
            'RightUpperLeg','RightLowerLeg','RightFoot','RightToe',
            'LeftUpperLeg','LeftLowerLeg','LeftFoot','LeftToe',
            'RightUpperArm','RightForeArm','RightHand',
            'LeftUpperArm','LeftForeArm','LeftHand',
            'Floor', 'Coordinate'
        ]

        self.gl_widget = openGLWidget(self, XSensVisualizer.WINDOW_WIDTH, XSensVisualizer.WINDOW_HEIGHT-100, self.mapped_segment_names)
        self.init_layout()

    def init_layout(self):
        root_layout = QVBoxLayout()
        root_layout.addWidget(self.gl_widget)
        self.setLayout(root_layout)

    def draw_model(self):
        for idx, key in enumerate(self.skeleton_data.keys()):
            self.gl_widget.updateData(idx, key, self.skeleton_data[key]['position'], self.skeleton_data[key]['quaternion'])
        self.gl_widget.display()

    def set_skeleton_segment_from_data_row(self, segment_name, data_row, center_offset=None):
        column_names = [v[0] + v[1] for v in itertools.product(['position_%s' % segment_name], self.xyz)]
        self.skeleton_data[segment_name]['position'] = np.array(data_row[column_names].values, dtype=float)
        if center_offset is not None:
            self.skeleton_data[segment_name]['position'] -= center_offset

    def set_skeleton_quaternion_from_data_row(self, segment_name, data_row):
        column_names = [v[0] + v[1] for v in itertools.product(['orientation_%s' % segment_name], self.quaternions)]
        self.skeleton_data[segment_name]['quaternion'] = np.array(data_row[column_names].values, dtype=float)

    def update_model(self, data_row):

        # --- Load raw positions and orientations for feet + toes first ---
        for seg in ["RightFoot", "LeftFoot", "RightToe", "LeftToe"]:
            self.set_skeleton_segment_from_data_row(seg, data_row)
            self.set_skeleton_quaternion_from_data_row(seg, data_row)

        # --- Compute center offset using TOE positions ---
        center_pos_offset = np.array([
            (self.skeleton_data['RightToe']['position'][0] +
            self.skeleton_data['LeftToe']['position'][0]) * 0.5,
            (self.skeleton_data['RightToe']['position'][1] +
            self.skeleton_data['LeftToe']['position'][1]) * 0.5,
            0.0
        ])

        # --- Apply center offset to feet + toes ---
        for seg in ["RightFoot", "LeftFoot", "RightToe", "LeftToe"]:
            self.skeleton_data[seg]['position'] -= center_pos_offset

        # --- Load and center all remaining segments ---
        for key in self.skeleton_data:
            if key not in ["RightFoot", "LeftFoot", "RightToe", "LeftToe"]:
                self.set_skeleton_segment_from_data_row(key, data_row, center_pos_offset)
                self.set_skeleton_quaternion_from_data_row(key, data_row)

        # --- Draw the updated skeleton ---
        self.draw_model()
