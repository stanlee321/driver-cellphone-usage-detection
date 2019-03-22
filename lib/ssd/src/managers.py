import multiprocessing
import numpy as np

from .object_detection.utils import visualization_utils as vis_util


class OutputFilter:
    """
    This class is used for deal with the ouputs of the detections
    """
    def __init__(self):
        self.category_index = None
    def grab_draws(self, image_np, detections):
        if  detections is not None:
            boxes = detections['boxes']
            classes = detections['classes']
            scores = detections['scores']
            num = detections['num']

            if (num > 0):
                print('------------')
                print('num', num)
                print('classes=', classes[0])
                print('scores=', scores[0])

                # top_k = results.argsort()[-4:][::-1]
                for i in range(num):
                    if classes[0][i] != 1:
                        classes[0][i] = 1
                    else:
                        #classes[0][i] = classes[0][i] #+ 1.0
                        idd = int(classes[0][i])
                        sco = scores[0][i]
                        name = 'none'
                        if (idd in self.category_index):
                            s = self.category_index[idd]
                            name = s['name']
                        print(name, '/', sco)

                # Visualization of the results of a detection.
                image_np = vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    min_score_thresh=0.52,
                    max_boxes_to_draw=num,
                    line_thickness=2)
        
        return image_np
