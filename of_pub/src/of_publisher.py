#!/usr/bin/env python

# Python includes
import numpy as np
import cv2
import time

# ROS includes
import rospy
from of_pub.msg import of_msg

def opticalflow_publisher(u, v):
    pub = rospy.Publisher('opticalflow', of_msg, queue_size=10)
    rospy.init_node('opticalflow_publisher', anonymous=True)

    data = of_msg()
    data.u = u
    data.v = v
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

lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 10,
                    qualityLevel = 0.3,
                    minDistance = 10,
                    blockSize = 7 )

def optical_flow(idx):
    trajectory_len = 40
    detect_interval = 5
    trajectories = []
    frame_idx = 0
    cap = cv2.VideoCapture(idx)

    score = 0

    # while True:
    now = time.time()
    while time.time() < now+50:
        # start time to calculate FPS
        start = time.time()
        suc, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = frame.copy()
        img = Rotate(img,180) # Rotate 

        u = []
        v = []

        # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
        if len(trajectories) > 0:
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1

            new_trajectories = []

            # Get all the trajectories
            for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                trajectory.append((x, y))
                if len(trajectory) > trajectory_len:
                    del trajectory[0]
                new_trajectories.append(trajectory)
                # Newest detected point
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

            trajectories = new_trajectories

            # Draw all the trajectories
            cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
            cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)

            left_sc = 0
            go_sc = 0

            for trajectory in trajectories:
                xzero = trajectory[0][0]
                yzero = trajectory[0][1]
                xprime = trajectory[-1][0]
                yprime = trajectory[-1][1]
                left_sc += xprime - xzero
                go_sc += yprime - yzero
                u.append(xprime - xzero)
                v.append(yprime - yzero)
            score += left_sc
            # if left_sc > 0 and go_sc > 0 :
            #     # print("LEFT")
            #     print("RIGHT")
            # elif left_sc <0 and abs(left_sc) > abs(go_sc):
            #     # print("RIGHT")
            #     print("LEFT")
            # elif go_sc > 0 and abs(go_sc) > abs(left_sc):
            #     # print("BACKWARD")
            #     print("GO")
            # else :
            #     # print("GO")
            #     print("BACKWARD")

            opticalflow_publisher(u,v)

            
        # Update interval - When to update and detect new features
        if frame_idx % detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255

            # Lastest point in latest trajectory
            for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
                cv2.circle(mask, (x, y), 5, 0, -1)

            # Detect the good features to track
            p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
            if p is not None:
                # If good features can be tracked - add that to the trajectories
                for x, y in np.float32(p).reshape(-1, 2):
                    trajectories.append([(x, y)])

        frame_idx += 1
        prev_gray = frame_gray

        # End time
        end = time.time()
        # calculate the FPS for current frame detection
        fps = 1 / (end-start)
        
        # Show Results
        cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Optical Flow', img)
        cv2.imshow('Mask', mask)
        # cv2.waitKey(0)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            pass

    print(score/10000)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    idx = find_snapbotcamidx()
    optical_flow(idx[-1])