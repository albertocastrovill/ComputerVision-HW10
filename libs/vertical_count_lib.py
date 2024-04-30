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


#Function to count the number of times the person is on the right side of the frame

def vertical_line_cout(vertical_frame):
    draw_vertical_line(vertical_frame)

    