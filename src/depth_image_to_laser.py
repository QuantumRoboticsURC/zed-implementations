#!/usr/bin/env python3

import time
import numpy as np
import cv2

gray_img = cv2.imread('./images/grayscale_img.jpeg', cv2.IMREAD_GRAYSCALE)


def depth_image_to_laser_scan(gray_img, n):
    
    height = gray_img.shape[0]
    width = gray_img.shape[1]
    
    # Initialize an empty array
    laser_scan = np.zeros((width,), dtype=np.float32)
    
    # Iterate over each column of the image
    for i in range(width):
        # Get the pixel values for the current column
        column = gray_img[:, i]
        
        # Sort the pixel values in ascending order
        sorted_column = np.sort(column)[:n]
        
        # Compute the average of the n smallest pixel values
        avg = np.mean(sorted_column)
        
        # Add the average value to the laser scan data
        laser_scan[i] = avg
    
    return laser_scan


def get_smallest_values_per_column(gray_img):
    
    height, width = gray_img.shape
    smallest_values = np.zeros((height, width))

    for x in range(width):
        column = gray_img[:, x]
        sorted_column = np.sort(column)
        smallest_values[:, x] = sorted_column[:height]

    return smallest_values

if __name__ == "__main__":
    print("resultado\n", depth_image_to_laser_scan(gray_img, 1))
    print("smallest values \n", get_smallest_values_per_column(gray_img))