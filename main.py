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
import libs.orb_detection as lo  

def run_pipeline(video_path, image_path, max_distance):
    # Initialize video capture
    cap = vl.open_video(video_path)
    if cap is None:
        print("Failed to open video.")
        return

    # Load the reference image and initialize ORB
    ref_image = lo.load_image(image_path)
    if ref_image is None:
        print("Failed to load reference image.")
        return

    orb= lo.initialize_orb(ref_image)
    ref_kp, ref_desc = lo.detect_and_compute(orb, ref_image)

    while True:
        frame = vl.read_frame(cap)
        if frame is None:
            break
        
        #Show original frame
        vl.show_frame("Original Video", frame)

        # Process and show vertical line frame
        vertical_frame = vcl.draw_vertical_line(frame.copy())
        #vl.show_frame("Vertical Video", vertical_frame)

        # Process and show diagonal line frame
        #diagonal_frame = dcl.draw_diagonal_line(frame.copy())
        #vl.show_frame("Diagonal Video", diagonal_frame)

        # ORB detection
        #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_desc = lo.detect_and_compute(orb, frame.copy())
        matches = lo.match_features(ref_desc, frame_desc)
        centroid = lo.compute_centroid(frame_kp, matches)
        filtered_matches = lo.filter_matches_by_centroid(matches, frame_kp, centroid, max_distance)
        matched_frame = lo.draw_matches(ref_image, ref_kp, frame, frame_kp, filtered_matches, centroid, max_distance)
        vertical_matched_frame = lo.draw_matches(ref_image, ref_kp, vertical_frame, frame_kp, filtered_matches, centroid, max_distance)
        vl.show_frame("ORB Matches", matched_frame)
        vl.show_frame("ORB Matches Vertical", vertical_matched_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    vl.close_video(cap)

if __name__ == "__main__":
    image_path = "videos/tello.jpg"
    video_path = "videos/TelloVideo3.mp4"
    run_pipeline(video_path, image_path, 50)
