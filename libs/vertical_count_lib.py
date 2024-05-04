# Esta librería va a recibir un video y va a contar cuantas veces se mueve la persona detectada en el video de izquierda a derecha.

# Import the necessary libraries
import cv2
import numpy as np

# Function to draw a vertical line in the middle of the frame
def draw_vertical_line(vertical_frame):
    # Get the dimensions of the frame
    height, width, _ = vertical_frame.shape
    # Draw a vertical line in the middle of the frame
    cv2.line(vertical_frame, (width//2, 0), (width//2, height), (0, 255, 0), 2)
    return vertical_frame

# Function to count the number of times the person is on the left side of the frame
def count_left_side(vertical_frame):
    # Get the dimensions of the frame
    height, width, _ = vertical_frame.shape
    # Get the left side of the frame
    left_side = vertical_frame[:, :width//2]
    # Convert the left side to grayscale
    left_side_gray = cv2.cvtColor(left_side, cv2.COLOR_BGR2GRAY)
    # Apply a threshold to the left side
    _, left_side_thresh = cv2.threshold(left_side_gray, 127, 255, cv2.THRESH_BINARY)
    # Find the contours in the left side
    contours, _ = cv2.findContours(left_side_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Count the number of contours in the left side
    count = len(contours)
    return count

#Function to count the number of times the person is on the right side of the frame
def count_right_side(vertical_frame):
    # Get the dimensions of the frame
    height, width, _ = vertical_frame.shape
    # Get the right side of the frame
    right_side = vertical_frame[:, width//2:]
    # Convert the right side to grayscale
    right_side_gray = cv2.cvtColor(right_side, cv2.COLOR_BGR2GRAY)
    # Apply a threshold to the right side
    _, right_side_thresh = cv2.threshold(right_side_gray, 127, 255, cv2.THRESH_BINARY)
    # Find the contours in the right side
    contours, _ = cv2.findContours(right_side_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Count the number of contours in the right side
    count = len(contours)
    return count

def vertical_line_cout(vertical_frame):
    draw_vertical_line(vertical_frame)

    