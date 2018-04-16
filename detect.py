import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import scipy.misc

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
import pickle
import time
import track

sys.path.append("object_detection")

from utils import label_map_util

MODEL_NAME = 'faster_rcnn_resnet101_coco_2017_11_08'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'object_detection/'+ MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, \
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR_ROOT = 'track1_frames/'
PATH_TO_TEST_IMAGES_DIRS = os.listdir(PATH_TO_TEST_IMAGES_DIR_ROOT)

for TEST_IMAGES_DIR in PATH_TO_TEST_IMAGES_DIRS:

	print(TEST_IMAGES_DIR)
	
	if TEST_IMAGES_DIR.startswith('.'):
		continue

	PATH_TO_TEST_IMAGES_DIR = PATH_TO_TEST_IMAGES_DIR_ROOT + TEST_IMAGES_DIR + '/'
	TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1,1801) ]


	with detection_graph.as_default():
	    with tf.Session(graph=detection_graph) as sess:
	        # Definite input and output Tensors for detection_graph
	        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	        # Each box represents a part of the image where a particular object was detected.
	        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	        # Each score represent how level of confidence for each of the objects.
	        # Score is shown on the result image, together with the class label.
	        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
	        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
	        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
	        
	        all_boxes=[]
	        all_scores=[]
	        all_classes=[]
	        all_images=[]
	        
	        count = 0
	        total = len(TEST_IMAGE_PATHS)
	        for image_path in TEST_IMAGE_PATHS:
	            since = time.time()
	            count += 1
	            image = Image.open(image_path)
	            # the array based representation of the image will be used later in order to prepare the
	            # result image with boxes and labels on it.
	            image_np = load_image_into_numpy_array(image)
	            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	            image_np_expanded = np.expand_dims(image_np, axis=0)
	            # Actual detection.
	            (boxes, scores, classes, num) = sess.run(
	            [detection_boxes, detection_scores, detection_classes, num_detections],
	            feed_dict={image_tensor: image_np_expanded})
	            
	            all_boxes.append(boxes)
	            all_scores.append(scores)
	            all_classes.append(classes)
	#             all_images.append(image_np)
	            print('{}/{}  {}  {}'.format(count,total,time.time()-since,TEST_IMAGES_DIR))
	            
	pickle.dump( (all_boxes,all_scores,all_classes), open( "all_p/detect_p/"+TEST_IMAGES_DIR+".p", "wb" ) )