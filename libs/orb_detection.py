import cv2
import numpy as np

def load_image(image_path):
    """Load the reference image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error loading image")
        return None
    return image

def initialize_orb(image):
    """Initialize ORB detector and matcher."""
    orb = cv2.ORB_create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE, WTA_K=2, scaleFactor=1.5)

    return orb

def detect_and_compute(orb, image):
    """Detect keypoints and compute descriptors for the image."""
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    """Match features between two sets of descriptors."""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    # Aplicar el test de ratio
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:  # Asegúrate de que hay dos coincidencias
            m, n = m_n
            if m.distance < 0.68 * n.distance:
                good_matches.append(m)
    return good_matches

"""
def draw_matches2(img1, kp1, img2, kp2, matches):
    #Draw matches between two images.
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return result
"""

def draw_matches(img1, kp1, img2, kp2, matches, centroid, max_distance):
    """Draw matches and the centroid area of interest."""
    # Dibujar las coincidencias
    output_image = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

    if centroid is not None:
        # Convertir las coordenadas del centroide a enteros
        centroid = (int(centroid[0]), int(centroid[1]))

        # Dibujar el centroide
        cv2.circle(output_image, centroid, 5, (255, 0, 0), -1)  # Punto azul para el centroide

        # Dibujar un círculo alrededor del centroide que define el área de interés
        cv2.circle(output_image, centroid, max_distance, (0, 255, 0), 2)  # Círculo verde para el área de interés

    return output_image



def compute_centroid(keypoints, matches):
    """Compute the centroid of matched keypoints."""
    if not matches:
        return None  # No matches to compute centroid
    points = np.array([keypoints[m.trainIdx].pt for m in matches])
    if points.size == 0:
        return None  # No points to calculate centroid
    centroid = np.mean(points, axis=0)
    return centroid

def filter_matches_by_centroid(matches, keypoints, centroid, max_distance):
    """Filter matches to keep only those close to the centroid."""
    good_matches = []
    for match in matches:
        pt = keypoints[match.trainIdx].pt
        if np.linalg.norm(np.array(pt) - centroid) < max_distance:
            good_matches.append(match)
    return good_matches
