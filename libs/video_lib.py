# Import libraries to desplay the video
import cv2

# Function to display the video
def display_video(video_path):
    # Read the video
    cap = cv2.VideoCapture(video_path)
    # Check if the video was opened
    if not cap.isOpened():
        print("Error opening video stream or file")
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            cv2.imshow('Video', frame)
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    return


