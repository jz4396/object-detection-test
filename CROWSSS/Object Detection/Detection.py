from Object_detection import Objdet
import cv2

test = Objdet("..\..\object_detection")

test.downloadGraph("ssd_mobilenet_v1_coco_11_06_2017","..\..\object_detection\data\mscoco_label_map.pbtxt",90)

test.listCategories()

test.live_Detection(0)