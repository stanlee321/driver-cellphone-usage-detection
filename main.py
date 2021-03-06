import cv2 # For webcam
import sys
import os

from lib.ssd.ssd_processor import SSDProcessor

IM_WIDTH = 640
IM_HEIGHT = 480

detect = SSDProcessor()
detect.setup()


camera = cv2.VideoCapture(0) # For custom video input, replace this 0 with a string with the 'name of your video.mp4'

if ((camera == None) or (not camera.isOpened())):
    print('\n\n')
    print('Error - could not open video device.')
    print('\n\n')
    exit(0)

ret = camera.set(cv2.CAP_PROP_FRAME_WIDTH,IM_WIDTH)
ret = camera.set(cv2.CAP_PROP_FRAME_HEIGHT,IM_HEIGHT)

# save the actual dimensions
actual_video_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_video_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

print('actual video resolution: ' + str(actual_video_width) + ' x ' + str(actual_video_height))

# Initialize frame rate calculation
frame_rate_calc = 1

freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

frame_count = 0
while(True):
    t1 = cv2.getTickCount()
    
    for i in range(5):
        camera.grab()

    ret, frame = camera.read()

    frame_count += 1
    
    detection = detect.detect(frame)

    boxes = detection['boxes']
    scores = detection['scores']
    classes = detection['classes']
    num = detection['num']

    print('frame:', frame_count)
    cv2.putText(frame,"FPS: {0:.2f} frame: {1}".format(frame_rate_calc, frame_count),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
    # All the results have been drawn on the frame, so it's time to display it.
    frame = detect.annotate_image(frame, boxes, classes, scores)
    cv2.imshow('Object detector', frame)
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    frame_count += 1
camera.release()

cv2.destroyAllWindows()