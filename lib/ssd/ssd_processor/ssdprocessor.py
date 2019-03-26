import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
from PIL import Image
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import logging
from pathlib import Path
import click


class SSDProcessor(object):
    """
        Performs object detection on an image
    """
    PATH_TO_MODEL = os.path.join(os.path.dirname(__file__),'..',
                        'model',
                        'ssdlite_mobilenet_v2_coco',
                        'frozen_inference_graph.pb')

    PATH_TO_LABELS = os.path.join(os.path.dirname(__file__),
                        'object_detection',
                        'data',
                        'mscoco_label_map.pbtxt')

    LINK_TO_DOWNLOAD_MODEL = "http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz"
    
    def __init__(self, path_to_model=None, path_to_labels=None, model_name=None):
        """
        Path to frozen detection graph. This is the actual model that is used for the object detection.
        strings used to add correct label for each box.
        """
        self._path_to_model = self.PATH_TO_MODEL
        self._path_to_labels = self.PATH_TO_LABELS
        self._num_classes = 90
        self._detection_graph = None
        self._labels = dict()
        self._image = None
        self._boxes = None
        self._classes = None
        self._scores = None
        self._num = None
        self._logger = None
        self._session = None
        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        self.num_detections = None

        self.index_to_string = {
            3: 'car',
            6: 'bus',
            8: 'truck',
            1: 'person',
            10: 'traffic light'
        }

    def setup(self):
        """
            Setup the model, if model does not exist in path, this will be downloaded
        """
        if not Path(self._path_to_model).exists():

            message = 'no object detection model available, would you like to download the model? download will take approx 100mb of space'
            if click.confirm(message):
                #self.download_model(self.LINK_TO_DOWNLOAD_MODEL)
                pass

        # Load model
        self.load_model(self._path_to_model)
        
        # Load labels
        self._labels = self.load_labels(self._path_to_labels)

        # run a detection once, because first model run is always slow
        self.detect(np.ones((300, 300, 3), dtype=np.uint8))

    def download_model(self, url):
        """
            Download a model file from the url and unzip it
            :url = Link where is the model .tar.gz
        """
        _filename = 'ssdlite_mobilenet_v2_coco'

        opener = urllib.request.URLopener()
        opener.retrieve(url, _filename)
        """
        tar_file = tarfile.open(_filename)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, path=str(Path(self._path_to_model).parents[1]))
        """

    def load_model(self, path):
        """
            Load saved model from protobuf file
            :path = Path to the model, String
        """
        if not Path(path).exists():
            raise IOError('model file missing: {}'.format(str(path)))
 
        with tf.gfile.GFile(path, 'rb') as fid:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fid.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        self._detection_graph = graph
        self._session = tf.Session(graph=self._detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self._detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.

        self.detection_scores = self._detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self._detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self._detection_graph.get_tensor_by_name('num_detections:0')

    def load_labels(self, path):
        """
            Load labels from .pb file, and map to a dict with integers, e.g. 1=aeroplane
            :path = Path to the Labels, String
        """

        label_map = label_map_util.load_labelmap(path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self._num_classes,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def load_image_into_numpy_array(self, path, scale=1.0):
        """
            Load image from disk into NxNx3 numpy array
        """
        image = Image.open(path)
        image = image.resize(tuple(int(scale * dim) for dim in image.size))
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def detect(self, image):
        """
            Detect objects in the image
            :image = Image array, numpy array

            :return: list of the detection outputs.
        """

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)

        # Actual detection.
        (self._boxes, self._scores, self._classes, self._num) = self._session.run([self.detection_boxes, 
                                                                            self.detection_scores,
                                                                            self.detection_classes, 
                                                                            self.num_detections],
                                            feed_dict={self.image_tensor: image_np_expanded})
        prediction = {
            'boxes': self._boxes,
            'classes': self._classes,
            'scores': self._scores,
            'num': self._num
        }
        return prediction


    def annotate_image(self, image, boxes, classes, scores, threshold=0.5):
        """
            Draws boxes around the detected objects and labels them

        :image: image array
        :boxes: list with annotation coord.
        :classes: number of classes to annotate.
        :score: score outputs.
        :theshhold: number of prob to filter above of this.

        :return: annotated image
        """
        
        annotated_image = image.copy()
        vis_util.visualize_boxes_and_labels_on_image_array(
            annotated_image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self._labels,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=threshold)
        
        return annotated_image

    def filter_class(self, detections, class_to_filter):

        data["predictions"] = []
        # loop over the results and add them to the list of
        # returned predictions
        # Filter just car detections.
        if num > 0:
            for i in range(num): #enumerate(boxes[0]):
                classes[0][i] = classes[0][i] + 1               # Added + 1.0  for lite compat.
                _id = int(classes[0][i])
                if _id == class_to_filter:
                    if scores[0][i] >= 0.1:
                        x0 = int(boxes[0][i][3] * image.shape[1])
                        y0 = int(boxes[0][i][2] * image.shape[0])

                        x1 = int(boxes[0][i][1] * image.shape[1])
                        y1 = int(boxes[0][i][0] * image.shape[0])

                        if draw_box is True:
                            color_map = {}

                            # assign color
                            r_color = lambda: random.randint(0, 255)
                            
                            for i in range(len(boxes[0])):
                                color_map[i] = (r_color(), r_color(), r_color(), 90)

                            draw = ImageDraw.Draw(pil_image, mode="RGBA")
                            
                            draw.rectangle((x0, y0, x1, y1),  outline=color_map[i], fill=color_map[i])
                            #draw.text((x0, y0),
                            #          ObjectDetection.index_to_string[int(classes[0][i])],
                            #          font=ImageFont.truetype("arial"))
                            # TODO draw text labels into the detection image
                        r = {
                            'image': draw,
                            'coord': {
                                'xmin': x0, 'ymin': y0,
                                'xmax': x1, 'ymax': y1
                            },
                            'class': model.labels[_id]['name'],          #ObjectDetection.index_to_string[id],
                            'probability': float(scores[0][i])
                        }
                        data["success"] = True
                        data["predictions"].append(r)
                    else:
                        pass
                else:
                    pass
    @property
    def labels(self):
        return self._labels

    def close(self):
        self._session.close()
