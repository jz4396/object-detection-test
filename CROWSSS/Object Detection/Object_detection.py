import cv2 as cv

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import time



class Objdet:
    """description of class"""
    def __init__(self, API_PATH):
        
        #Append path of object detection folder so you can import the utils
        sys.path.append(API_PATH)
        sys.path.append(API_PATH + "/../")

        #import Api utils
        global label_map_util
        from utils import label_map_util

        global vis_util
        from utils import visualization_utils as vis_util

    def downloadGraph(self, MODEL_NAME, LABEL_PATH, NUM_CLASSES):
        #specify models to download from tensorflow
        MODEL_FILE = MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
        CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        #download and extract 
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

        self.loadGraph(CKPT, LABEL_PATH, NUM_CLASSES)

    def loadGraph(self, PATH_TO_CKPT, LABEL_PATH, NUM_CLASS):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        
        PATH_TO_LABELS = os.path.join(LABEL_PATH)
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASS, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        self.sess = tf.Session(graph=self.detection_graph)

    def listCategories(self):
        print("\n\nCategories(ID:Name):")
        for id, name in self.category_index.items():
            print(id,': ', name['name'])


    def cam_Detection(self, cap):
        if(cap.isOpened()):
            ret, self.image = cap.read()
            return self.overlay_Detection(self.image)
    
    def rpi_cam_Detection(self, camera, rawCapture):
        camera.capture(rawCapture, format="bgr")
        self.image = self.rawCapture.array
        rawCapture.truncate(0)
        return self.overlay_Detection(self.image)


    def run_Detection(self, image_np):
        with self.detection_graph.as_default():
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (self.boxes, self.scores, self.classes, self.num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={image_tensor: image_np_expanded})

    def overlay_Detection(self, image_np):
        self.run_Detection(image_np)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(self.boxes),
            np.squeeze(self.classes).astype(np.int32),
            np.squeeze(self.scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        return image_np

    def live_Detection(self, CAM_NUM, tim=False,WIN_NAME=""):
        cap = cv.VideoCapture(CAM_NUM)
        if tim:
            timely = timer(True)
        while cv.waitKey(1) not in [27,ord('q')]:
        #for i in range(250):
            try:
                cv.imshow(WIN_NAME, self.cam_Detection(cap))
            except:
                cv.destroyAllWindows()
                print("Detection Failed")
                return -1
            if tim:
                timely.toc(True)
        cv.destroyAllWindows()
        if tim:
            print("Average: "+str(timely.average()))

    def rpi_live_Detection(self, tim=False, resolution=(1920,1080), WIN_NAME=""):
        from picamera.array import PiRGBArray
        from picamera import PiCamera

        camera = PiCamera()
        rawCapture = PiRGBArray(camera)
        camera.resolution = resolution
        if tim:
            timely = timer(True)
        while cv.waitKey(1) not in [27,ord('q')]:
            try:
                cv.imshow(WIN_NAME, self.rpi_cam_Detection(camera, rawCapture))
            except:
                cv.destroyAllWindows()
                print("Detection Failed")
                return -1
            if tim:
                timely.toc(True)
        cv.destroyAllWindows()
        if tim:
            print("Average: "+str(timely.average()))


class timer:
    def __init__(self,log = False):
        self.times=[]
        self.start=time.time()
        self.isLogging = log
    def tic(self):
        self.start = time.time()
    def toc(self, printer = False):
        timely = time.time()-self.start
        if printer:
            print(timely)
        if self.isLogging:
            self.times.append(timely)
        self.start = time.time()
        return timely
    def average(self):
        return np.mean(self.times)