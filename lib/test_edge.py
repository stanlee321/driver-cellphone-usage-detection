import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(
        description='This sample shows how to define custom Opencv2 deep learning layers in Python. '
                    'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                    'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--width', help='Resize input image to a specific width', default=256, type=int)
parser.add_argument('--height', help='Resize input image to a specific height', default=256, type=int)
args = parser.parse_args()


"""
Test with:

python test_edge.py \
    --input 'test_img.png' \
    --width 640 --height 480
"""

def main():

    # load the input image and grab its dimensions
    image = cv2.imread(args.input)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # convert the image to grayscale, blur it, and perform Canny
    # edge detection
    print("[INFO] performing Canny edge detection...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 30, 150)

    out=cv2.cvtColor(canny,cv2.COLOR_GRAY2BGR)
    out = out.astype(np.uint8)

    con=np.concatenate((image,out, hsv_img),axis=1)
    cv2.imshow("kWinName", con)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()