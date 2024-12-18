# Import libraries
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import glob

# Set main directory
data_dir = '/Users/afonso/Documents/LAB_PROJETO/Projeto final/Dataset1/'

# Set the coordinates of the LANE area
lane_coordinates = [(100, 500), (400, 500), (600, 720), (0, 720)]

# Function to check if a point is within the LANE area
def point_in_lane(point):
    x, y = point
    return (
        (x > lane_coordinates[0][0] and x < lane_coordinates[1][0] and y > lane_coordinates[0][1] and y < lane_coordinates[2][1]) or
        (x > lane_coordinates[1][0] and x < lane_coordinates[2][0] and y > lane_coordinates[1][1] and y < lane_coordinates[3][1])
    )

# Prepare image lists
raw_images_list = []
lane_images_list = []
crack_images_list = []
pothole_images_list = []

# Go through the subfolders
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)

    # Check if it's a valid folder
    if os.path.isdir(folder_path):
        # Load the RAW image
        raw_images_list.extend(glob.glob(os.path.join(folder_path, '*RAW.jpg')))

        # Load the LANE mask
        lane_images_list.extend(glob.glob(os.path.join(folder_path, '*LANE.png')))

        # Load the CRACK mask
        crack_images_list.extend(glob.glob(os.path.join(folder_path, '*CRACK.png')))

        # Load the POTHOLE mask
        pothole_images_list.extend(glob.glob(os.path.join(folder_path, '*POTHOLE.png')))

# Ensure the lists are in correct order
raw_images_list.sort()
lane_images_list.sort()
crack_images_list.sort()
pothole_images_list.sort()

# Initialize index
idx = 0

# Loop to navigate through images
while True:
    # Load raw image and masks
    raw_image = cv2.imread(raw_images_list[idx])
    lane_mask = cv2.imread(lane_images_list[idx], cv2.IMREAD_GRAYSCALE)
    _, lane_mask = cv2.threshold(lane_mask, 127, 255, cv2.THRESH_BINARY)
    crack_mask = cv2.imread(crack_images_list[idx], cv2.IMREAD_GRAYSCALE)
    _, crack_mask = cv2.threshold(crack_mask, 127, 255, cv2.THRESH_BINARY)
    pothole_mask = cv2.imread(pothole_images_list[idx], cv2.IMREAD_GRAYSCALE)
    _, pothole_mask = cv2.threshold(pothole_mask, 127, 255, cv2.THRESH_BINARY)

    # Apply masks and find contours
    lane_cracks = cv2.bitwise_and(crack_mask, lane_mask)
    lane_potholes = cv2.bitwise_and(pothole_mask, lane_mask)

    # Use bitwise operation to isolate CRACKS from the overlap
    isolated_lane_cracks = cv2.bitwise_and(lane_cracks, cv2.bitwise_not(lane_potholes))

    # Find the contours of the lane holes
    crack_contours, _ = cv2.findContours(isolated_lane_cracks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pothole_contours, _ = cv2.findContours(lane_potholes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image (Pothole contours are drawn first)
    cv2.drawContours(raw_image, pothole_contours, -1, (0, 255, 255), 3)
    cv2.drawContours(raw_image, crack_contours, -1, (0, 0, 255), 3)

    # Add the captions in the bottom left corner of the image
    cv2.putText(raw_image, 'Rachadura', (10, raw_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(raw_image, 'Buraco', (10, raw_image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Display the image with the lane holes and captions
    cv2.imshow('Imagem com buracos da lane', raw_image)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):  # Quit
        break
    elif key == ord('n'):  # Next image
        idx = (idx + 1) % len(raw_images_list)  # Loop back to the beginning if at end
    elif key == ord('p'):  # Previous image
        idx = (idx - 1) % len(raw_images_list)  # Loop back to the end if at the beginning

cv2.destroyAllWindows()
