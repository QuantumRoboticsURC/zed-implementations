#!/usr/bin/env python3
import rospy
import pyzed.sl as sl
import math
import numpy as np
import sys
import cv2


def main():

    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)



    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
    # Setting the depth confidence parameters
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.textureness_confidence_threshold = 100

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Capture images and depth while ros is running
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m

    while not rospy.is_shutdown():
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)




            image_ocv = image_zed.get_data()
            depth_image_ocv = depth_image_zed.get_data()
            point_cloud_ocv = point_cloud.get_data()
            #print ("Image Size: ", image_ocv.shape)
            #print ("Depth Image Size: ", depth_image_ocv.shape)


            # Get and print distance value in mm at the center of the image
            # We measure the distance camera - object using Euclidean distance
            x = round(image_zed.get_width() / 2)
            y = round(image_zed.get_height() / 2)
            err, point_cloud_value = point_cloud.get_value(x, y)
            # Point Cloud
            xCloud = point_cloud_value[0]
            yCloud = point_cloud_value[1]
            zCloud = point_cloud_value[2]
            colorCloud = point_cloud_value[3]

            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                 point_cloud_value[1] * point_cloud_value[1] +
                                 point_cloud_value[2] * point_cloud_value[2])


            point_cloud_np = point_cloud.get_data()
            point_cloud_np.dot(tr_np)
             
            # Radius of circle
            radius = 10
            
            # Blue color in BGR
            color = (0, 255, 0)
            
            # Line thickness of 2 px
            thickness = 2            
            image_ocv = cv2.circle(image_ocv, (x,y), radius, color, thickness)

            
            cv2.imshow("Image", image_ocv)
            #cv2.imshow("Depth", depth_image_ocv)

            key = cv2.waitKey(10)
            if not np.isnan(distance) and not np.isinf(distance):
                print("")
                print("Posicion X: ", x)
                print("Posicion Y: ", y)
                print("Distancia: ", distance)
                print("Cloud en X: ", xCloud)
                print("Cloud en Y: ", yCloud)
                print("Cloud en Z: ", zCloud)

            else:
                print("Can't estimate distance at this position.")
                print("Your camera is probably too close to the scene, please move it backwards.\n")
            sys.stdout.flush()
            rate.sleep()


    # Close the camera
    zed.close()
    print("\nFINISH")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass