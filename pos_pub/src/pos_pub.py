#!/usr/bin/env python

# Python includes
import cv2
import numpy as np
import time
import math
import apriltag
import pyrealsense2 as rs
from pos_pub.class_motionhelper import timer
from pos_pub.utils_track import tps_trans, get_tps_mat

# ROS includes
import rospy
from std_msgs.msg import String

def apriltag_publisher(x_pos, y_pos, yaw, Hz, LOG_INFO = True):
    pub = rospy.Publisher('apriltag_position', String, queue_size=10)
    rospy.init_node('apriltag_publisher', anonymous=True)
    rate = rospy.Rate(Hz)

    pos = "%s %s %s" % (x_pos, y_pos, yaw)
    if LOG_INFO:rospy.loginfo(pos)
    pub.publish(pos)
    rate.sleep()

def publish_xy(tps_coef, Hz, max_sec, LOG_INFO, VERBOSE=False):
    real_xy_y_traj = np.empty(shape=(0,2))
    x = np.linspace(170,470,4)
    y = np.linspace(90,390,4)
    X, Y = np.meshgrid(x,y)
    ctrl_xy = np.stack([X,Y],axis=2).reshape(-1,2)
    real_center_pos = np.zeros([1,2])

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    time.sleep(1)
    idx, flag, threshold = 0, False, 0
    t = timer(_HZ=Hz, _MAX_SEC=max_sec)
    t.start()
    while t.is_notfinished():
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        results = detector.detect(gray)

        for r in results:
            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            # draw the bounding box of the AprilTag detection
            if VERBOSE : 
                cv2.line(color_image, ptA, ptB, (0, 255, 0), 2)
                cv2.line(color_image, ptB, ptC, (0, 255, 0), 2)
                cv2.line(color_image, ptC, ptD, (0, 255, 0), 2)
                cv2.line(color_image, ptD, ptA, (0, 255, 0), 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            center_pos = np.array([[cX, cY]])
            real_center_pos = tps_trans(center_pos, ctrl_xy, tps_coef)
            if VERBOSE : 
                cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)
            tany = abs((ptC[1]+ptD[1])/2 - cY)
            tanx = (ptC[0]+ptD[0])/2 - cX
            rad = math.atan2(tanx, tany)
            deg = int(rad * 180 / math.pi)
            real_xy_y_traj = np.append(real_xy_y_traj, np.array([[real_center_pos[0, 1], -real_center_pos[0, 0]]]), axis=0)
            apriltag_publisher(real_xy_y_traj[-1, 1], -real_xy_y_traj[-1, 0], rad, Hz, LOG_INFO)
            
            if VERBOSE: 
                cv2.putText(color_image, "({},{}),{}".format(cX, cY, deg), (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if VERBOSE: 
            cv2.imshow("Frame", color_image)
            if cv2.waitKey(20) == 27:
                break   
    if VERBOSE:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    tps_coef = get_tps_mat()
    publish_xy(tps_coef, 50, 600, LOG_INFO = True, VERBOSE = True)