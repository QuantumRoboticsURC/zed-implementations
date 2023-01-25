#!/usr/bin/env python3

import pyzed.sl as sl
import numpy as np
import argparse
import time
import cv2
import sys
import math
import pandas as pd

class ExperimentHelper():
    def __init__(self, aruco_dict = cv2.aruco.DICT_4X4_50):

        # ________ aruco atributes initialization ______
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
        time.sleep(1.0)
        self.zed_runtime_parameters = sl.RuntimeParameters()
        self.zed_runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
        self.zed_runtime_parameters.confidence_threshold = 100
        self.zed_runtime_parameters.textureness_confidence_threshold = 100
        self.image_size = self.zed_camera.get_camera_information().camera_resolution

        # TODO: IMPORTANT!!!! DEFINE THE IMAGE SIZE THAT WE WILL BE USING AT URC, SO THAT 
        # VALUES IN EXPERIMENTS CORRESPOND TO THE VALUES AT COMPETITION
        # =========================================================================
        self.image_size.width = 640
        self.image_size.height = 360
        # =========================================================================

        self.image_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)
        self.depth_image_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)
        self.point_cloud = sl.Mat()

        self.imge_ocv = np.zeros((self.image_size.height, self.image_size.width, 3), dtype=np.uint8)
        self.depth_image_zed_ocv = np.zeros((self.image_size.height, self.image_size.width), dtype=np.uint8)
        self.point_cloud_ocv = np.zeros((self.image_size.height, self.image_size.width), dtype=np.uint8)
        self.displayed_image_ocv = np.zeros((self.image_size.height, self.image_size.width, 3), dtype=np.uint8)
        self.arucos_mask = np.zeros((self.image_size.height, self.image_size.width, 3), dtype = np.uint8)
        self.arucos_mask_with_distance = np.zeros((self.image_size.height, self.image_size.width), dtype = np.float64)

        # ______________ experiment data initialization ______________
        self.experiment_data = { "aruco x center px":[],
                            "aruco y center px":[],
                            "aruco z center px":[],
                            "aruco area px":[],
                            "aruco top left corner x px":[],
                            "aruco top left corner y px":[],
                            "aruco top right corner x px":[],
                            "aruco top right corner y px":[],
                            "aruco bottom right corner x px":[],
                            "aruco bottom right corner y px":[],
                            "aruco bottom left corner x px":[],
                            "aruco bottom left corner y px":[],
                            "aruco x center meters":[],
                            "aruco y center meters":[],
                            "aruco z center meters":[],
                            "aruco rot roll (deg)":[],
                            "aruco rot pitch (deg)":[],
                            "aruco rot yaw (deg)":[]
                            }

    def draw_arucos(self, image, corners):
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
        return (corners, ids)

    def midpoint_equation(self, p1, p2):
        return ( (p1[0]+p2[0])/2, (p1[1]+p2[1])/2 )

    def get_aruco_midpoint_and_area(self, rectangle_corners):
        """ function that returns the x,y,z cordinates of the aruco's midpoint """
        # ______________ initializing and formating data that will be used ______________
        self.arucos_mask = np.zeros((self.image_size.height, self.image_size.width, 3), dtype = np.uint8)
        rectangle_corners_for_x_y = rectangle_corners.reshape((4,2))
        rectangle_corners_for_mask = np.int32(rectangle_corners.reshape((1,4,2)))
        # ______________ getting the x,y cordinates of the aruco tag (in pixels) ______________
        x_center, y_center = self.midpoint_equation(rectangle_corners_for_x_y[0,:], rectangle_corners_for_x_y[2,:])
        # ______________ getting the z cordinate of the aruco tag (in point cloud units) ______________
        # step one - we filter the point cloud using a mask with only the area of the aruco tag
        cv2.fillPoly(self.arucos_mask, pts = rectangle_corners_for_mask, color=(255,255,255))  
        one_channel_arucos_mask = cv2.cvtColor(self.arucos_mask, cv2.COLOR_BGR2GRAY)  /255.0        
        self.arucos_mask_with_distance = np.nan_to_num(self.point_cloud_ocv)*one_channel_arucos_mask
        # step two - we get the mean point cloud value on the aruco tag area, to use it as z value
        tag_area = one_channel_arucos_mask.sum()
        if tag_area > 0:        
            z_center = (self.arucos_mask_with_distance/255.0).sum()/tag_area
        else:
            z_center = 0.0
        return (float(x_center), float(y_center), float(z_center), float(tag_area))

    def input_is_valid(self, given_input):
        try:
            value = eval(given_input)
            if len(value) == 6:
                if type(value[0]) is float:
                    return True
            return False
        except:
            return False

    def main(self):
        user_input = ""            
        df = pd.read_csv('./experiments_data/experiment_1.csv', index_col=False)

        while True:            
            print("please give me the current artag pose, using the folloeing format")
            print("[x_position, y_position, z_position, roll_deg, pitch_deg, yaw_deg]")            
            user_input = input("POSE:")
            self.zed_camera.retrieve_image(self.image_zed, sl.VIEW.LEFT, sl.MEM.CPU, self.image_size)
            self.zed_camera.retrieve_image(self.depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, self.image_size)
            self.zed_camera.retrieve_measure(self.point_cloud, sl.MEASURE.DEPTH, sl.MEM.CPU, self.image_size)
            self.image_ocv = self.image_zed.get_data()
            self.image_ocv = self.image_ocv[:,:,:-1]
            self.depth_image_ocv = self.depth_image_zed.get_data()
            self.point_cloud_ocv = self.point_cloud.get_data()
            
            self.displayed_image_ocv = self.image_ocv.copy()
            self.arucos_mask = np.zeros((self.image_size.height, self.image_size.width, 3), dtype = np.int8)
            self.arucos_mask_with_distance = np.zeros((self.image_size.height, self.image_size.width), dtype = np.float64)
            
            if user_input == "end":
                break
            elif self.input_is_valid(user_input) and self.zed_camera.grab(self.zed_runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                aruco_corners, aruco_ids = self.get_arucos_info_in_image(self.image_ocv)
                aruco_centers_and_areas = list(map(self.get_aruco_midpoint_and_area, aruco_corners))
                aruco_pose = eval(user_input)

                if len(aruco_centers_and_areas) > 0:
                    aruco_corner = aruco_corners[0].reshape((4,2)) # reshape the corners of the first aruco in the image
                    self.experiment_data["aruco x center px"].append(aruco_centers_and_areas[0][0])
                    self.experiment_data["aruco y center px"].append(aruco_centers_and_areas[0][1])
                    self.experiment_data["aruco z center px"].append(aruco_centers_and_areas[0][2])
                    self.experiment_data["aruco area px"].append(aruco_centers_and_areas[0][3])
                    self.experiment_data["aruco top left corner x px"].append(aruco_corner[0,0])
                    self.experiment_data["aruco top left corner y px"].append(aruco_corner[0,1])
                    self.experiment_data["aruco top right corner x px"].append(aruco_corner[1,0])
                    self.experiment_data["aruco top right corner y px"].append(aruco_corner[1,1])
                    self.experiment_data["aruco bottom right corner x px"].append(aruco_corner[2,0])
                    self.experiment_data["aruco bottom right corner y px"].append(aruco_corner[2,1])
                    self.experiment_data["aruco bottom left corner x px"].append(aruco_corner[3,0])
                    self.experiment_data["aruco bottom left corner y px"].append(aruco_corner[3,1])
                    self.experiment_data["aruco x center meters"].append(aruco_pose[0])
                    self.experiment_data["aruco y center meters"].append(aruco_pose[1])
                    self.experiment_data["aruco z center meters"].append(aruco_pose[2])
                    self.experiment_data["aruco rot roll (deg)"].append(aruco_pose[3])
                    self.experiment_data["aruco rot pitch (deg)"].append(aruco_pose[4])
                    self.experiment_data["aruco rot yaw (deg)"].append(aruco_pose[5])

                    self.displayed_image_ocv = self.draw_arucos(self.displayed_image_ocv, aruco_corners)
                    cv2.imshow("image", self.displayed_image_ocv)                
                    cv2.waitKey(0)
            else:
                print("invalid input!")
        df = df.append(pd.DataFrame(self.experiment_data))
        df.to_csv('./experiments_data/experiment_1.csv', index=False)

if __name__ == "__main__":
    aruco_detector = ExperimentHelper()
    aruco_detector.main()
