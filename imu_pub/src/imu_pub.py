#!/usr/bin/env python

# Python includes
import os
import serial

# ROS includes
import rospy
from std_msgs.msg import String

usb = 0
for i in range(10):
    if os.path.exists("/dev/ttyUSB{}".format(i)):
        usb = i
os.system("sudo chmod a+rw /dev/ttyUSB{}".format(usb))

ser = serial.Serial('/dev/ttyUSB{}'.format(usb), 115200, timeout=1)

def imu_publisher(Hz=100):
    pub = rospy.Publisher('imu_sensor', String, queue_size=10)
    rospy.init_node('imu_publisher', anonymous=True)
    rate = rospy.Rate(Hz)

    while not rospy.is_shutdown():
        line = ser.readline()
        data = line.decode('unicode_escape')
        rospy.loginfo(data)
        pub.publish(data)
        rate.sleep

if __name__ == '__main__':
    try:
        imu_publisher()
    except rospy.ROSInterruptException: pass

    ser.close()