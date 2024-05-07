"""
This code is the main pipeline for the project. It reads a video file and an image file, and detects the position of the
drone in the video. It uses ORB feature detection and matching to detect the image in the video frames, and then uses the
centroid of the matched points to determine the position of the drone. The code also counts the number of times the drone
crosses a vertical line in the video, and displays the count on the video frames.

Author: Alberto Castro
Date: 06/04/2024

Usage:
    python3 main.py

"""

# Import necessary libraries
import os
import cv2
import numpy as np
import libs.video_lib as vl
import libs.vertical_count_lib as vcl
import libs.orb_detection as lo


def run_pipeline(video_path, image_path, max_distance, min_matches):
    """
    Run the main pipeline for the project.

    Args:
        video_path (str): Path to the video file.
        image_path (str): Path to the image file.
        max_distance (int): Maximum distance between matched points to consider a valid match.
        min_matches (int): Minimum number of matches required to consider the image detected.

    Returns:
        None
    """
    
    cap = vl.open_video(video_path)
    if cap is None:
        print("Failed to open video.")
        return

    ref_image = lo.load_image(image_path)
    if ref_image is None:
        print("Failed to load reference image.")
        return

    orb = lo.initialize_orb(ref_image)
    ref_kp, ref_desc = lo.detect_and_compute(orb, ref_image)

    last_centroid = None
    centroid_history = []  # History to smooth centroid movement
    history_length = 10  # Number of past centroids to consider for smoothing
    last_position = None
    left_count = 0
    right_count = 0
    last_change_time = 0
    min_time_between_changes = 1000  # Min time between changes in milliseconds
    initial_position_counted = False

    while True:
        frame = vl.read_frame(cap)
        if frame is None:
            break

        vertical_frame = vcl.draw_vertical_line(frame.copy())
        mid_line_x = frame.shape[1] // 2

        frame_kp, frame_desc = lo.detect_and_compute(orb, vertical_frame)
        matches = lo.match_features(ref_desc, frame_desc)

        current_time = cv2.getTickCount() / cv2.getTickFrequency() * 1000  # Current time in milliseconds

        if len(matches) > min_matches:
            current_centroid = lo.compute_centroid(frame_kp, matches)
            if current_centroid is not None:
                centroid_history.append(current_centroid)
                if len(centroid_history) > history_length:
                    centroid_history.pop(0)

                smoothed_centroid = np.mean(centroid_history, axis=0) if centroid_history else current_centroid
                current_position = 'left' if smoothed_centroid[0] < mid_line_x else 'right'

                if not initial_position_counted:
                    if current_position == 'left':
                        left_count += 1
                    else:
                        right_count += 1
                    initial_position_counted = True

                if last_centroid is not None and last_position != current_position:
                    if (current_time - last_change_time > min_time_between_changes):
                        if current_position == 'left':
                            left_count += 1
                        else:
                            right_count += 1
                        last_change_time = current_time

                last_centroid = smoothed_centroid
                last_position = current_position

        # Draw centroid and maximum distance circle if exists
        if last_centroid is not None:
            center = (int(last_centroid[0]), int(last_centroid[1]))
            cv2.circle(vertical_frame, center, 5, (255, 0, 0), -1)
            cv2.circle(vertical_frame, center, max_distance, (0, 255, 0), 2)

        # Display counts
        cv2.putText(vertical_frame, f"Left: {left_count}", (10, vertical_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vertical_frame, f"Right: {right_count}", (vertical_frame.shape[1] - 200, vertical_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Video with Line and Centroid", vertical_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    vl.close_video(cap)

if __name__ == "__main__":
    image_path = "videos/tello.jpg"
    video_path = "videos/TelloVideo3.mp4"
    run_pipeline(video_path, image_path, 80, 13)




