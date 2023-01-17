#!/usr/bin/env python3

# import the necessary packages
# import imutils
# from imutils.video import VideoStream
import rospy
import pyzed.sl as sl
import numpy as np
import argparse
import time
import cv2
import sys
import math
from std_msgs.msg import String

def main():	
	rospy.init_node('dete_aruco', anonymous=True)
	# Create a Camera object
	zed = sl.Camera()
	debug_topic = rospy.Publisher("/debug_print", String, queue_size=1)


	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	# default="DICT_ARUCO_ORIGINAL",
	ap.add_argument("-t", "--type", type=str,
		default="DICT_4X4_50",
		help="type of ArUCo tag to detect")
	args = vars(ap.parse_args())

	# define names of each possible ArUco tag OpenCV supports
	ARUCO_DICT = {
		"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
		"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
		"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
		"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
		"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
		"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
		"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
		"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
		"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
		"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
		"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
		"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
		"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
		"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
		"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
		"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
		"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
		"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
		"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
		"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
		"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
	}

	# verify that the supplied ArUCo tag exists and is supported by
	# OpenCV
	# Lines 43-46 check to see if the ArUco tag --type exists in our ARUCO_DICT. If not, we exit the script.

	if ARUCO_DICT.get(args["type"], None) is None:
		print("[INFO] ArUCo tag of '{}' is not supported".format(
			args["type"]))
		sys.exit(0)

	# load the ArUCo dictionary and grab the ArUCo parameters
	print("[INFO] detecting '{}' tags...".format(args["type"]))
	arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
	arucoParams = cv2.aruco.DetectorParameters_create()

	# Create a InitParameters object and set configuration parameters
	init_params = sl.InitParameters()
	init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use PERFORMANCE depth mode
	init_params.camera_resolution = sl.RESOLUTION.HD720

	# Open the camera
	err = zed.open(init_params)
	if err != sl.ERROR_CODE.SUCCESS:
		exit(1)

	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")

	# vs = VideoStream(src=0).start()
	time.sleep(2.0)

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

	# Capture images while ros is running
	image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
	depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
	point_cloud = sl.Mat()

	# debug_topic.publish("{t}".format(t=type(image_zed)))

	mirror_ref = sl.Transform()
	mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
	tr_np = mirror_ref.m

	# loop over the frames from the video stream
	while not rospy.is_shutdown():
		if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
	        # Retrieve left image
			zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
	        # Retrieve depth map. Depth is aligned on the left image
			zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
	        # Retrieve colored point cloud. Point cloud is aligned on the left image.
			zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)
			image_ocv = image_zed.get_data()
			image_ocv = np.uint8(image_ocv)
			image_ocv = image_ocv[:,:,:-1]
			depth_image_ocv = depth_image_zed.get_data()
			point_cloud_ocv = point_cloud.get_data()

			debug_topic.publish("{t}, size: {s}".format(t=type(image_ocv), s=image_ocv.shape))

	        # Get and print distance value in mm at the center of the image
	        # We measure the distance camera - object using Euclidean distance
			x = round(image_zed.get_width() / 2)
			y = round(image_zed.get_height() / 2)
			err, point_cloud_value = point_cloud.get_value(x, y)
	            # Point Cloud
			xCloud = point_cloud_value[0]
			yCloud = point_cloud_value[1]
			zCloud = point_cloud_value[2]
			# colorCloud = point_cloud_value[3]

			distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
								 point_cloud_value[1] * point_cloud_value[1] +
								 point_cloud_value[2] * point_cloud_value[2])


			point_cloud_np = point_cloud.get_data()
			point_cloud_np.dot(tr_np)

		# detect ArUco markers in the input frame
		(corners, ids, rejected) = cv2.aruco.detectMarkers(image_ocv,
			arucoDict, parameters=arucoParams)

		final_img = image_ocv.copy()
		# verify *at least* one ArUco marker was detected
		if len(corners) > 0:
			# flatten the ArUco IDs list
			ids = ids.flatten()

			# loop over the detected ArUCo corners
			for (markerCorner, markerID) in zip(corners, ids):
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
				cv2.line(final_img, topLeft, topRight, (0, 255, 0), 2)
				cv2.line(final_img, topRight, bottomRight, (0, 255, 0), 2)
				cv2.line(final_img, bottomRight, bottomLeft, (0, 255, 0), 2)
				cv2.line(final_img, bottomLeft, topLeft, (0, 255, 0), 2)

				# compute and draw the center (x, y)-coordinates of the
				# ArUco marker
				cX = int((topLeft[0] + bottomRight[0]) / 2.0)
				cY = int((topLeft[1] + bottomRight[1]) / 2.0)
				cv2.circle(final_img, (cX, cY), 4, (0, 0, 255), -1)

				# draw the ArUco marker ID on the frame
				cv2.putText(final_img, str(markerID),
					(topLeft[0], topLeft[1] - 15),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5, (0, 255, 0), 2)

		# show the output frame
		cv2.imshow("Image", final_img)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanupimage_ocv =
	cv2.destroyAllWindows()
	zed.close()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
