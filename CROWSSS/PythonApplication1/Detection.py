from Object_detection import Objdet, timer
import cv2

test = Objdet("..\..\object_detection")

test.downloadGraph("ssd_mobilenet_v1_coco_11_06_2017","..\..\object_detection\data\mscoco_label_map.pbtxt",90)

if(test.live_Detection(0)==-1):
    cv2.imshow("failure",test.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()