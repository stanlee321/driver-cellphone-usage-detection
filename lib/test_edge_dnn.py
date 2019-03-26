import cv2
import argparse
import numpy as np
import time
from edge.edge_detection import CropLayer

parser = argparse.ArgumentParser(
        description='This sample shows how to define custom Opencv2 deep learning layers in Python. '
                    'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                    'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--prototxt', help='Path to deploy.prototxt', required=True)
parser.add_argument('--caffemodel', help='Path to hed_pretrained_bsds.caffemodel', required=True)
parser.add_argument('--width', help='Resize input image to a specific width', default=256, type=int)
parser.add_argument('--height', help='Resize input image to a specific height', default=256, type=int)
args = parser.parse_args()

"""
Test with:

python test_edge.py \
    --input '/home/stanlee321/dwhelper/out.mp4' \
    --prototxt edge/hed_model/deploy.prototxt \
    --caffemodel edge/hed_model/hed_pretrained_bsds.caffemodel \
    --width 640 --height 480
"""
def main():
    cv2.dnn_registerLayer('Crop', CropLayer)

    # Load the model.
    net = cv2.dnn.readNet(args.prototxt, args.caffemodel)

    kWinName = 'Holistically-Nested_Edge_Detection'
    cv2.namedWindow(kWinName, cv2.WINDOW_AUTOSIZE)

    cap = cv2.VideoCapture(args.input if args.input else 0)

    while True:
        t1 = time.time()
        _, frame = cap.read()

        frame = cv2.resize(frame, (640,480))


        inp = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(args.width, args.height),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
        net.setInput(inp)
        
        out = net.forward()

        out = out[0, 0]

        out = cv2.resize(out, (frame.shape[1], frame.shape[0]))

        out=cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)

        out = 255 * out

        con=np.concatenate((frame,out),axis=1)
        t2 = time.time()

        print('TIME IS::', t2 -t1)
        cv2.imshow(kWinName, con)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()