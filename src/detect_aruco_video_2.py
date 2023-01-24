#!/usr/bin/env python3

"""Made by:Erika García Sánchez
	A01745158@tec.mx
	erika.mgs@outlook.com
	José Ángel del Ángel
    joseangeldelangel10@gmail.com

Modified (15/12/2022): 
		José Ángel del Ángel and Erika García 16/12/2022 Aruco detection code cleanup 
		José Ángel del Ángel 16/12/2022 Aruco mask with distance added

Code description:
TODO - update code description and notes 
1. Ask the user for the name of a file.
2. If the file exist at a predefined directory it opens it and read it.
   If not, it creates a file with the name.
3. Ask the user the action to execute (write lat & long | write the square).
4. Close the file. 

Notes:
- Validate the user input
- Add an exit option
* Despite the code adds 0.00001 theorically, it's not exact.
"""
import rospy
import pyzed.sl as sl
import numpy as np
import argparse
import time
import cv2
import sys
import math
from std_msgs.msg import String, Int8

class ArucoDetector():
    def __init__(self, aruco_dict = cv2.aruco.DICT_4X4_50):
        rospy.init_node("aruco_detector")

        # ________ aruco atributes initialization ______
        #self.arucoDict = cv2.aruco.Dictionary_get(aruco_dict)
        self.arucoDict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.arucoDetector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

        # ________ camera atributes initialization ______
        self.zed_camera = sl.Camera()
        self.zed_init_params = sl.InitParameters()
        self.zed_init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use PERFORMANCE depth mode
        self.zed_init_params.camera_resolution = sl.RESOLUTION.HD720
        err = self.zed_camera.open(self.zed_init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)
        rospy.sleep(2.0)
        self.zed_runtime_parameters = sl.RuntimeParameters()
        self.zed_runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
        self.zed_runtime_parameters.confidence_threshold = 100
        self.zed_runtime_parameters.textureness_confidence_threshold = 100
        self.image_size = self.zed_camera.get_camera_information().camera_resolution
        self.image_size.width = self.image_size.width /2
        self.image_size.height = self.image_size.height /2

        self.image_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)
        # TODO: delete depth_image_zed after debugging
        self.depth_image_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)
        self.point_cloud = sl.Mat()
        self.imge_ocv = np.zeros((self.image_size.height, self.image_size.width, 3), dtype=np.uint8)
        self.depth_image_zed_ocv = np.zeros((self.image_size.height, self.image_size.width), dtype=np.uint8)
        self.point_cloud_ocv = np.zeros((self.image_size.height, self.image_size.width), dtype=np.uint8)
        self.displayed_image_ocv = np.zeros((self.image_size.height, self.image_size.width, 3), dtype=np.uint8)
        self.arucos_mask = np.zeros((self.image_size.height, self.image_size.width, 3), dtype = np.uint8)
        self.arucos_mask_with_distance = np.zeros((self.image_size.height, self.image_size.width), dtype = np.float32)

        # ________ ros atributes initialization ______
        self.debug_topic = rospy.Publisher("/debug_print", String, queue_size=1)
        self.aruco_distances_publisher = rospy.Publisher("/aruco_distances", String, queue_size = 1)

    def draw_arucos(sel, image, corners):
        # verify *at least* one ArUco marker was detected
        if len(corners) > 0:
			# loop over the detected ArUCo corners
            for markerCorner in corners:
				# extract the marker corners (which are always returned
				# in top-left, top-right, bottom-right, and bottom-left
				# order)
                corners = markerCorner.reshape((4, 2))

                (topLeft, topRight, bottomRight, bottomLeft) = corners

				# convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

				# draw the bounding box of the ArUCo detection
                image = cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                image = cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                image = cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                image = cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
        return image

    def get_arucos_info_in_image(self, image):
        # detect ArUco markers in the input frame
        (corners, ids, rejected) = self.arucoDetector.detectMarkers(image)
        # self.debug_topic.publish("aruco corners : {c}, aruco corners dtype {t}".format(c = corners, t = type(corners)))
        return (corners, ids)

    def midpoint_equation(self, p1, p2):
        return ( (p1[0]+p2[0])/2, (p1[1]+p2[1])/2 )

    def get_aruco_midpoint(self, rectangle_corners):
        self.arucos_mask = np.zeros((self.image_size.height, self.image_size.width, 3), dtype = np.uint8)
        rectangle_corners_for_x_y = rectangle_corners.reshape((4,2))
        rectangle_corners_for_mask = np.int32(rectangle_corners.reshape((1,4,2)))
        x_center, y_center = self.midpoint_equation(rectangle_corners_for_x_y[0,:], rectangle_corners_for_x_y[2,:])
        cv2.fillPoly(self.arucos_mask, pts = rectangle_corners_for_mask, color=(255,255,255))
        one_channel_arucos_mask = cv2.cvtColor(self.arucos_mask, cv2.COLOR_BGR2GRAY)  /255.0
        print( "one channel mask maximmum is: {m} ".format(m = one_channel_arucos_mask.max()) ) # TODO: delete this
        self.arucos_mask_with_distance = self.point_cloud_ocv*one_channel_arucos_mask
        tag_area = one_channel_arucos_mask.sum()
        if tag_area > 0.0:        
            z_center = (self.arucos_mask_with_distance/255.0)/one_channel_arucos_mask.sum()
        else:
            z_center = 0.0
        return (float(x_center), float(y_center), float(z_center) )

    def main(self):
        while not rospy.is_shutdown():
            self.arucos_mask = np.zeros((self.image_size.height, self.image_size.width, 3), dtype = np.int8)
            self.arucos_mask_with_distance = np.zeros((self.image_size.height, self.image_size.width), dtype = np.float32)
            if self.zed_camera.grab(self.zed_runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve left image
                self.zed_camera.retrieve_image(self.image_zed, sl.VIEW.LEFT, sl.MEM.CPU, self.image_size)
                # Retrieve depth map. Depth is aligned on the left image
                self.zed_camera.retrieve_image(self.depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, self.image_size)
                # Retrieve colored point cloud. Point cloud is aligned on the left image.
                self.zed_camera.retrieve_measure(self.point_cloud, sl.MEASURE.DEPTH, sl.MEM.CPU, self.image_size)
                self.image_ocv = self.image_zed.get_data()
                self.image_ocv = self.image_ocv[:,:,:-1]
                self.depth_image_ocv = self.depth_image_zed.get_data()
                self.point_cloud_ocv = self.point_cloud.get_data()

                aruco_corners, aruco_ids = self.get_arucos_info_in_image(self.image_ocv)
                self.debug_topic.publish("aruco corners : {c}, aruco corners dtype {t}".format(c = aruco_corners, t = type(aruco_corners)))
                self.displayed_image_ocv = self.image_ocv.copy()
                self.displayed_image_ocv = self.draw_arucos(self.displayed_image_ocv, aruco_corners)                
                aruco_centers = list(map(self.get_aruco_midpoint, aruco_corners))
                self.debug_topic.publish("aruco_centers: {c}".format(c = aruco_centers))
                cv2.imshow("image", self.displayed_image_ocv)
                cv2.imshow("aruco mask", self.arucos_mask)
                cv2.imshow("aruco mask with depth", np.uint8(self.arucos_mask_with_distance))
                cv2.waitKey(1)                
                # self.displayed_image_ocv = self.image_ocv.copy()

if __name__ == "__main__":
    aruco_detector = ArucoDetector()
    aruco_detector.main()
