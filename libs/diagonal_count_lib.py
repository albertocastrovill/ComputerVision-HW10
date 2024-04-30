# Import the necessary libraries
import cv2
import numpy as np

# Function to draw a vertical line in the middle of the frame
def draw_diagonal_line(frame):
    # Get the dimensions of the frame
    height, width, _ = frame.shape
    # Draw a diagonal line form the bottom left to the top right of the frame
    cv2.line(frame, (0, height), (width, 0), (0, 255, 0), 2)
    
    return frame