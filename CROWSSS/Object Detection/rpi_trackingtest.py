from Object_detection import Objdet, timer
import cv2

#import RPi.GPIO as io
#io.setmode(io.BOARD)

test = Objdet("..\..\object_detection")

test.loadGraph("ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb","..\..\object_detection\data\mscoco_label_map.pbtxt",90)

from picamera.array import PiRGBArray
from picamera import PiCamera

camera = PiCamera()
rawCapture = PiRGBArray(camera)
camera.resolution = resolution

#yaw = io.PWM(12)
#yaw.start(50)

for i in range(250):
    camera.capture(rawCapture, "rgb")
    test.run_Detection(rawCapture.array())
    for i in enumerate(test.boxes[0]):
        if test.classes[0][i]==1:
            mid = (test.boxes[0][i][3]-test.boxes[0][i][1])/2
            print(mid)

#yaw.stop()