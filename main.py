"""
This script will have the main function to run the program, it will call other programs to run this program.

Author: Alberto Castro, Ana Bárbara Quintero, Hector Camacho
Date: 2024-04-30

"""

# Importing the necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import libs.video_lib as vl


if __name__ == "__main__":
    vl.display_video("videos/Video1_HW10.mp4")