from Object_detection import Objdet, timer
import cv2

test = Objdet("..\..\object_detection")

test.loadGraph("ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb","..\..\object_detection\data\mscoco_label_map.pbtxt",90)

test.live_Detection(0, True)

test.rpi_live_Detection(True)

from picamera.array import PiRGBArray
from picamera import PiCamera

camera = PiCamera()
rawCapture = PiRGBArray(camera)
camera.resolution = resolution

time_rpi = timer(True)
for i in range(250):
    camera.capture(rawCapture, "rgb")
    test.run_Detection(rawCapture.array())
    time_rpi.toc()

print(time_rpi.average)


time_cv = timer(True)
cam = cv2.VideoCapture(0)
for i in range(250):
    _, cap = cam.read()
    test.run_Detection(cap)
    time_cv.toc()

print(time_cv.average)