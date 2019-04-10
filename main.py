import cv2 # For webcam
import sys
import os
import pprint
from lib.ssd.ssd_processor import SSDProcessor
from src.sort import Sort

pp = pprint.PrettyPrinter(indent=4)

IM_WIDTH = 640
IM_HEIGHT = 480

# Detector
detect = SSDProcessor()
detect.setup()

min_score_thresh = 0.50
draw_box = True



# Tracker
tracker = Sort(use_dlib= False, max_age=5, min_hits=0) #create instance of the SORT tracker

src = '/home/stanlee321/dwhelper/out.mp4'

camera = cv2.VideoCapture(src) # For custom video input, replace this 0 with a string with the 'name of your video.mp4'

if ((camera == None) or (not camera.isOpened())):
    print('\n\n')
    print('Error - could not open video device.')
    print('\n\n')
    exit(0)

# Set input resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH,IM_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,IM_HEIGHT)

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
    
    frame = cv2.resize(frame, (320, 240))

    frame_count += 1
    
    detection = detect.detect(frame)

    boxes = detection['boxes']
    scores = detection['scores']
    classes = detection['classes']
    num = detection['num']

    # All the results have been drawn on the frame, so it's time to display it.
    #frame = detect.annotate_image(frame, boxes, classes, scores)
    
    detections, frame = detect.annotate_image_and_filter(frame,
                                                boxes, 
                                                classes, 
                                                scores, 
                                                num, 
                                                min_score_thresh, 
                                                draw_box)

    # The detections from the object detection algorithm mus be in the format:

    # dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]

    #update tracker with the detections...
    trackers = tracker.update(detections['association'], frame)

    #print('TRACKERS....', trackers)
    for d in trackers:
        print('TRACKED ::>>>',d)
    frame = detect.annodate_image_simple(frame, trackers)
    # pp.pprint(detections)

    # Draw some info
    cv2.putText(frame,"FPS: {0:.2f} frame: {1}".format(frame_rate_calc, frame_count),
            (30,50), 
            font, 
            0.5, 
            (255,255,0), 
            1, 
            cv2.LINE_AA)

    frame = cv2.resize(frame, (640, 480))

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