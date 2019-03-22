import cv2 # For webcam
import sys
import os
import argparse
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from image_processor import ImageProcessor

IM_WIDTH = 640
IM_HEIGHT = 480

# For pass argument file
parser = argparse.ArgumentParser(description='Add folder to process')
parser.add_argument('-f', '--checkImage', default = None, type=str, help="Add path to the folder to check")

args = parser.parse_args()

if args.checkImage != None:
    rutaDeTrabajo = args.checkImage
    print('Ruta a limpiar: {}'.format(rutaDeTrabajo))
else:
    print('No se introdujo folder a revisar')


if __name__ == '__main__':
    detect = ImageProcessor()
    detect.setup()
    fotografias = [f for f in os.listdir(rutaDeTrabajo) if '.jpg' in f]

    print('Analizando {} imagenes'.format(len(fotografias)))
    for fotografia in fotografias:
        path_to_original_image = rutaDeTrabajo+'/'+fotografia
        path_to_new_image = path_to_original_image[:path_to_original_image.rfind('.')]
        tiempoMedicion = time.time()
        frame = cv2.imread(path_to_original_image)
        frame = cv2.resize(frame,(IM_WIDTH, IM_HEIGHT))
        (boxes, scores, classes, num) = detect.detect(frame)
        print('>>>>>>>>>>>', time.time()-tiempoMedicion)
        frame = detect.annotate_image(frame, boxes, classes, scores)
        cv2.imwrite("{}_detected.jpg".format(path_to_new_image), frame)
        #print('RESULTS', classes)