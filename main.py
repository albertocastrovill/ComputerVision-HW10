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
import libs.vertical_count_lib as vcl
import libs.diagonal_count_lib as dcl

def run_pipeline(video_path):
    cap = vl.open_video(video_path)
    if cap is None:
        print("Failed to open video.")
        return

    while True:
        frame = vl.read_frame(cap)
        if frame is None:
            break
        
        #Show original frame
        vl.show_frame("Original Video", frame)

        # Process vertical line frame
        vertical_frame = vcl.draw_vertical_line(frame.copy())
        # Show vertical frame
        vl.show_frame("Vertical Video", vertical_frame)

        # Process diagonal line frame
        diagonal_frame = dcl.draw_diagonal_line(frame.copy())
        # Show diagonal frame
        vl.show_frame("Diagonal Video", diagonal_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    vl.close_video(cap)

if __name__ == "__main__":
    run_pipeline("videos/Video1_HW10.mp4")