import os
import cv2
import numpy as np
import libs.video_lib as vl
import libs.vertical_count_lib as vcl
import libs.orb_detection as lo

def run_pipeline(video_path, image_path, max_distance, min_matches):
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
    centroid_history = []  # For smoothing
    frames_since_last_update = 0
    history_length = 5  # Number of past centroids to consider for smoothing

    while True:
        frame = vl.read_frame(cap)
        if frame is None:
            break

        # Draw vertical line in the middle of the frame
        vertical_frame = vcl.draw_vertical_line(frame.copy())

        frame_kp, frame_desc = lo.detect_and_compute(orb, vertical_frame)
        matches = lo.match_features(ref_desc, frame_desc)

        if len(matches) > min_matches:
            current_centroid = lo.compute_centroid(frame_kp, matches)
            if current_centroid is not None:
                if last_centroid is not None:
                    distance = np.linalg.norm(np.array(current_centroid) - np.array(last_centroid))
                    if distance < max_distance:
                        centroid_history.append(current_centroid)
                        if len(centroid_history) > history_length:
                            centroid_history.pop(0)
                        last_centroid = np.mean(centroid_history, axis=0)
                        frames_since_last_update = 0
                else:
                    last_centroid = current_centroid
                    centroid_history.append(current_centroid)
                    frames_since_last_update = 0
        else:
            frames_since_last_update += 1

        # Reset centroid if no good updates for too long
        if frames_since_last_update > 10:
            last_centroid = None
            centroid_history.clear()
            frames_since_last_update = 0

        # Draw centroid and maximum distance circle if exists
        if last_centroid is not None:
            center = (int(last_centroid[0]), int(last_centroid[1]))
            cv2.circle(vertical_frame, center, 5, (255, 0, 0), -1)
            cv2.circle(vertical_frame, center, max_distance, (0, 255, 0), 2)

        cv2.imshow("Video with Line and Centroid", vertical_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    vl.close_video(cap)

if __name__ == "__main__":
    image_path = "videos/tello.jpg"
    video_path = "videos/TelloVideo3.mp4"
    run_pipeline(video_path, image_path, 80, 12)




