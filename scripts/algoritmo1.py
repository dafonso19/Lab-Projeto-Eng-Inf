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

# Ensure the lists are in the correct order
raw_images_list.sort()
lane_images_list.sort()
crack_images_list.sort()
pothole_images_list.sort()

# Initialize index
idx = 0

# Initialize a dictionary to hold counts of each hole type
hole_counts = {
    'Longitudinal': 0,
    'Transversal': 0,
    'Rachadura': 0,
}

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

    # Define colors for different hole types
    colors = {
        'Longitudinal': (0, 0, 255),  # Blue
        'Transversal': (0, 255, 0),  # Green
        'Rachadura': (255, 0, 0),  # Red
    }

    # Use bitwise operation to isolate CRACKS from the overlap
    isolated_lane_cracks = cv2.bitwise_and(lane_cracks, cv2.bitwise_not(lane_potholes))

    # Find the contours of the lane holes
    crack_contours, _ = cv2.findContours(isolated_lane_cracks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pothole_contours, _ = cv2.findContours(lane_potholes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contours in [pothole_contours, crack_contours]:
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            hole_type = 'Rachadura'  # Assume it's a crack by default
            if contour_area > 500:
                # Calculate the slope of the contour
                [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                slope = vy / vx

                # If the absolute slope is less than 1, the hole is longitudinal. Otherwise, it's transversal.
                if abs(slope) >= 1:
                    hole_type = 'Longitudinal'
                else:
                    hole_type = 'Transversal'

                # Draw contours on the raw image based on the hole type
                cv2.drawContours(raw_image, [contour], -1, colors[hole_type], 3)

                # Increase count for this hole type
                hole_counts[hole_type] += 1

            # Draw contours of cracks (rachaduras) in red color
            if hole_type == 'Rachadura':
                cv2.drawContours(raw_image, [contour], -1, colors['Rachadura'], 3)
                hole_counts['Rachadura'] += 1

    # Move captions to the top left corner of the image
    text_x = 10
    text_y = 20
    text_margin = 20
    text_line_height = 20
    
    # Define color and font size for the legend
    first_line_color = (0, 0, 0)  # Black color
    first_line_font_scale = 1.0  # Font scale factor for the first line
    font_scale = 0.6  # Font scale factor for the rest of the lines

    # Add the first line of the legend with custom color and font size
    cv2.putText(raw_image, 'Legend:', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, first_line_font_scale, first_line_color, 1)
    
    # Add the rest of the legend with the modified color and order
    cv2.putText(raw_image, 'Longitudinal: {}'.format(hole_counts['Longitudinal']), (text_x, text_y + text_margin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors['Longitudinal'], 2)
    cv2.putText(raw_image, 'Transversal: {}'.format(hole_counts['Transversal']), (text_x, text_y + 2 * text_margin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors['Transversal'], 2)
    cv2.putText(raw_image, 'Rachadura: {}'.format(hole_counts['Rachadura']), (text_x, text_y + 3 * text_margin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors['Rachadura'], 2)

    # Display the image
    cv2.imshow('Lane Detection', raw_image)

    # Wait for a key press
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):  # Quit
        break
    elif key == ord('n'):  # Next image
        idx = (idx + 1) % len(raw_images_list)  # Loop back to the beginning if at the end
    elif key == ord('p'):  # Previous image
        idx = (idx - 1) % len(raw_images_list)  # Loop back to the end if at the beginning

# Close the image display window
cv2.destroyAllWindows()
