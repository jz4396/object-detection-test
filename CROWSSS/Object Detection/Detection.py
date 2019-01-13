from Object_detection import Objdet
import cv2

test = Objdet("..\..\object_detection")

#test.downloadGraph("ssd_mobilenet_v1_coco_11_06_2017","..\..\object_detection\data\mscoco_label_map.pbtxt",90)
test.downloadGraph("ssd_mobilenet_v2_coco_2018_03_29","..\..\object_detection\data\mscoco_label_map.pbtxt",90)
#test.loadGraph("ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb","..\..\object_detection\data\mscoco_label_map.pbtxt",90)

test.listCategories()

test.live_Detection(0)

