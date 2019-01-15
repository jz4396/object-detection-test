from Object_detection import Objdet, timer
import cv2

test = Objdet("../object_detection")

test.loadGraph("ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb","../object_detection/data/mscoco_label_map.pbtxt",90)
print("_______________________________________________CV Test________________________________________________")
test.live_Detection(0, True)

print("_______________________________________________rpi Test________________________________________________")
test.rpi_live_Detection(True)

from picamera.array import PiRGBArray
from picamera import PiCamera

camera = PiCamera()
rawCapture = PiRGBArray(camera)
camera.resolution = resolution

print("______________________________________________RPI Displess test________________________________________________")
time_rpi = timer(True)
for i in range(250):
    camera.capture(rawCapture, "rgb")
    test.run_Detection(rawCapture.array())
    time_rpi.toc()
camera.close()
print(time_rpi.average)

print("_______________________________________________CV Displess test________________________________________________")
time_cv = timer(True)
cam = cv2.VideoCapture(0)
for i in range(250):
    _, cap = cam.read()
    test.run_Detection(cap)
    time_cv.toc()
cap.release()
print(time_cv.average)