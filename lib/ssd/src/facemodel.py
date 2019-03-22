import tensorflow as tf
import numpy as np
import os
import cv2
from .object_detection.utils import label_map_util


'''
tf_lite_cam.py
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/toco/g3doc/python_api.md
Created on 2018/09/04

TensorFlow Lite Python interpreter 
Using the interpreter from a model file 
'''

class FaceDetectionModule:
    """
    This class is used for instantiate the Face Detection model and Python Interpreter interface.
    """
    def __init__(self):
        
        BASE_DIR = os.getcwd().split('/')
        HOME = '/'+BASE_DIR[1]
        USER = BASE_DIR[2]
        ROOT_PROJECTS = BASE_DIR[3]
        LUCAM_APPS = 'lucam_apps'
        APP = 'face_detection'
        PATH = os.path.join(HOME, USER, ROOT_PROJECTS, LUCAM_APPS, APP)
        print(PATH)

        self.floating_model = False

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join(PATH,'model', 'face.pbtxt')
        NUM_CLASSES = 1
        tflite_models = ['face_detect_non_quant.tflite', 'face_detect_quant.tflite', 'face_detect_300x300_quant_08.tflite']

        print('PATH_TO_LABELS=', PATH_TO_LABELS)

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        # Load TFLite model and allocate tensors.
        PATH_TO_MODEL = os.path.join(PATH, 'model', tflite_models[-1])
        self.interpreter = tf.contrib.lite.Interpreter(model_path=PATH_TO_MODEL)
        self.interpreter.allocate_tensors()

        self.input_mean = 127.5
        self.input_std = 127.5

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # check the type of the input tensor
        if self.input_details[0]['dtype'] == np.float32:
            self.floating_model = True

        # NxHxWxC, H:1, W:2
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]


        print("height=", self.height)
        print("width=", self.width)

    @staticmethod
    def load_labels(filename):
        my_labels = []
        input_file = open(filename, 'r')
        for l in input_file:
            my_labels.append(l.strip())
        return my_labels
    
    def setup(self):
        # TODO, make the interpreter load here
        pass

    def predict(self, image_np):
        """
        This method is used for the inference
        """
        # Reshape to the expected input sizes
        image_np_x = cv2.resize(image_np, (self.height, self.width))

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        input_data_x = np.expand_dims(image_np_x, axis=0)
        # image_np_expanded = image_np_expanded.npresize((width, height))

        if self.floating_model:
            input_data = (np.float32(input_data_x) - self.input_mean) / self.input_std
        else:
            input_data = input_data_x
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # boxes, scores, classes, num
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        num = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])

        prediction = {
            'boxes': boxes,
            'classes': classes,
            'scores': scores,
            'num': num
        }
        return prediction
