"""
Usage:
    bag_to_mat.py <path>

Convert rosbag files to matlab MAT files
"""
from bagpy import bagreader
import os
import numpy as np
from docopt import docopt
from os import walk
import pandas as pd

if __name__ == "__main__":
    arguments = docopt(__doc__)
    mypath = arguments["<path>"]
    for filename in os.listdir(mypath):
        filepath = os.path.join(mypath, filename)
        if os.path.isfile(filepath):
            b = bagreader(filepath)

            IMU_MSG = b.message_by_topic("/hardware_a1/imu")
            POSE_MSG = b.message_by_topic("/hardware_a1/estimation_body_pose")
            JOINT_MSG = b.message_by_topic("/hardware_a1/joint_foot")
            CONTROL_MSG = b.message_by_topic("/a1_debug/control_output")