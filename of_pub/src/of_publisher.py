#!/usr/bin/env python

# Python includes
import numpy as np
import cv2
import time

# ROS includes
import rospy
from rospy.numpy_msg import numpy_msg
from of_pub.msg import of_msg

def opticalflow_publisher(u, v, shape):
    pub = rospy.Publisher('opticalflow', numpy_msg(of_msg), queue_size=10)
    rospy.init_node('opticalflow_publisher', anonymous=True)

    data = of_msg()
    data.u = u
    data.v = v
    data.shape = shape
    pub.publish(data)


def Rotate(src, degrees):
    if degrees == 90:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 1)

    elif degrees == 180:
        dst = cv2.flip(src, -1)

    elif degrees == 270:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 0)
    else:
        dst = src
    return dst

def find_snapbotcamidx():
    all_camera_idx_available = []
    for camera_idx in range(20):
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            all_camera_idx_available.append(camera_idx)
            cap.release()
        else : 
            pass
    return all_camera_idx_available         

def drawFlow(img,flow,step=16): # 16 pixel wide grid
    h,w = img.shape[:2]
    idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.int)
    indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)
    u = []
    v = []

    for x,y in indices:
        cv2.circle(img, (x,y), 1, (0,255,0), -1)
        dx,dy = flow[y, x].astype(np.int)
        cv2.line(img, (x,y), (x+dx, y+dy), (0,255, 0),2, cv2.LINE_AA )
        u.append(dx)
        v.append(dy)

    opticalflow_publisher(u,v,idx_x.shape)

def optical_flow(idx):
    prev = None # previous frame

    cap = cv2.VideoCapture(idx)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000/fps)

    while cap.isOpened():
        ret,frame = cap.read()
        frame = Rotate(frame,180)
        if not ret: break
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 

        if prev is None: 
            prev = gray
        else:
            flow = cv2.calcOpticalFlowFarneback(prev,gray,None,0.5,3,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN) 
            drawFlow(frame,flow)
            prev = gray
        
        cv2.imshow('OpticalFlow-Farneback', frame)
        if cv2.waitKey(delay) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    idx = find_snapbotcamidx()
    optical_flow(idx[-1])