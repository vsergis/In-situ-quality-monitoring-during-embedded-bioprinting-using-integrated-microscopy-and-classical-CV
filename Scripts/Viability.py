import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import measure

SourceF='./5ulLivedead_red'
File_name=os.listdir('./'+SourceF)

# Define HSV ranges for green and red colors
green_lower = np.array([35, 50, 50])  # Lower bound of green in HSV
green_upper = np.array([85, 255, 255])  # Upper bound of green in HSV

red_lower1 = np.array([0, 50, 50])  # Lower bound of red (1st range)
red_upper1 = np.array([10, 255, 255])  # Upper bound of red (1st range)
red_lower2 = np.array([170, 50, 50])  # Lower bound of red (2nd range)
red_upper2 = np.array([180, 255, 255])  # Upper bound of red (2nd range)

for ki in File_name:
    vname = SourceF+'/' + str(ki)
    tname = ki.split('.m')

    newpath = './5ulLivedead_results' + '/' + tname[0]

    # Load the RGB image
    image = cv2.imread(vname)  # Replace with your image path
    image_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Create masks for green and red cells
    green_mask = cv2.inRange(image_hsv, green_lower, green_upper)
    red_mask1 = cv2.inRange(image_hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(image_hsv, red_lower2, red_upper2)
    red_mask = red_mask1 | red_mask2  # Combine both red masks

    # Count the number of cells in each mask
    green_labels = measure.label(green_mask)
    red_labels = measure.label(red_mask)

    # Calculate total area covered by green and red cells
    green_area = np.sum(green_labels > 0)  # Total number of pixels labeled as green
    red_area = np.sum(red_labels > 0)  # Total number of pixels labeled as red

    # Calculate the area of each component
    green_props = measure.regionprops(green_labels)
    red_props = measure.regionprops(red_labels)

    # Sum the areas
    total_green_area = sum(prop.area for prop in green_props)
    total_red_area = sum(prop.area for prop in red_props)

    # Calculate cell viability based on area
    total_area = total_green_area + total_red_area
    if total_area > 0:
        viability = (total_green_area / total_area) * 100
    else:
        viability = 0


    # Create images for the green and red masks
    green_image = np.zeros_like(image_rgb)
    green_image[green_mask > 0] = image[green_mask > 0]  # Keep original BGR colors for green cells

    red_image = np.zeros_like(image_rgb)
    red_image[red_mask > 0] = image[red_mask > 0]  # Keep original BGR colors for red cells

    # Create a new blank image to place results
    output_height = max(image_rgb.shape[0], green_image.shape[0], red_image.shape[0])
    output_width = image_rgb.shape[1] * 3  # Combine images in a row
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # Place the images in the output image
    output_image[:image_rgb.shape[0], :image_rgb.shape[1]] = image  # Original image
    output_image[:green_image.shape[0],
    image_rgb.shape[1]:image_rgb.shape[1] + green_image.shape[1]] = green_image  # Green image
    output_image[:red_image.shape[0], image_rgb.shape[1] + green_image.shape[1]:] = red_image  # Red image

    # Add viability results text to the image
    text_lines = [
        f"Total area of green (live) cells: {total_green_area} pixels",
        f"Total area of red (dead) cells: {total_red_area} pixels",
        f"Cell Viability: {viability:.2f}%"
    ]

    # Set the starting position for the text
    y_start = output_height - 60
    for i, line in enumerate(text_lines):
        cv2.putText(output_image, line, (10, y_start + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    # Save the output image
    cv2.imwrite(newpath, output_image)
