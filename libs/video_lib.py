# Import libraries to desplay the video
import cv2
import numpy as np
from libs import vertical_count_lib as vcl

def open_video(video_path):
    """ Open the video file from the given path. """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return None
    return cap

def read_frame(cap):
    """ Read a frame from the video and display it. """
    ret, frame = cap.read()
    if ret:
        #cv2.imshow(window_name, frame)
        return frame
    return None

def show_frame(window_name, frame):
    """ Show the frame in a window. """
    cv2.imshow(window_name, frame)

def close_video(cap):
    """ Close the video file and destroy all windows. """
    cap.release()
    cv2.destroyAllWindows()


