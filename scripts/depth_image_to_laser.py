#!/usr/bin/env python3

import time
import numpy as np
import cv2
from time import process_time
from PIL import Image

gray_img = cv2.imread('./images/grayscale_img.jpeg', cv2.IMREAD_GRAYSCALE)


def np_image(width, height):
    img = np.ones((width, height), dtype=np.uint8)

    img[0:] = 0.0
    img[1:] = 255.0/2
    img[2:] = 255.0

    return img


def depth_image_to_laser_scan(gray_img, n):
    t1_start = process_time() 
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

    t1_stop = process_time()

    print("Elapsed time:", t1_stop, t1_start) 
    print("Elapsed time during the whole program in seconds:", t1_stop-t1_start),

    return laser_scan


def get_smallest_values_per_column(gray_img):
    
    height = gray_img.shape[0]
    width = gray_img.shape[1]
    smallest_values = np.zeros((width, ))

    for x in range(width):
        column = gray_img[:, x]
        smallest_values[x] = np.min(column)

    return smallest_values

def np_array_to_image(np_array):
    image = Image.fromarray(np.uint8(np_array))
    image.show()


if __name__ == "__main__":
    img_prueba = np_image(5,10)
    print("resultado\n", depth_image_to_laser_scan(img_prueba, 1))
    np_array = np.array([[0],[128],[255]])
    np_array_to_image(np_array)
    # print("smallest values \n", get_smallest_values_per_column(img_prueba))
    # cv2.imshow("image", img_prueba)
    # cv2.waitKey(0)
    