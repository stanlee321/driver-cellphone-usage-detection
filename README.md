# driver-cellphone-usage-detection

# Requirements

Install with pip

* filterpy
* tensorflow
* numba
* sklearn

The tracking algorithm is based on the official !(sort)[https://github.com/abewley/sort/]


We use this repo as a basis for the tracking:

https://github.com/ZidanMusk/experimenting-with-sort

This uses a modified version of ssd_mobilenetv2 trained on Kitti dataset for car and pedestrian detection.

The Sort tracking algorithm expects inputs from detection in the format:

`dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]`

# Get started

run 

```bash

$ python main.py
```

# Links

* https://arxiv.org/pdf/1312.6024.pdf
* http://www.robesafe.com/personal/javier.yebes/docs/Yebes10ectits.pdf
* https://www.groundai.com/project/real-time-distracted-driver-posture-classification/
* https://arxiv.org/pdf/1312.6024.pdf
* http://www.ee.iisc.ac.in/new/people/faculty/soma.biswas/Papers/bharadwaj_icgvip2016_vehicleData.pdf
* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.874.3672&rep=rep1&type=pdf
* file:///home/stanlee321/Downloads/059dd83807.pdf
* https://www.sentiance.com/2017/09/25/passenger-and-driver/
* https://arxiv.org/pdf/1903.04933.pdf
* https://ivrl.epfl.ch/research-2/research-downloads/supplementary_material-cvpr11-index-html/
* https://github.com/abewley/sort/issues/55
* https://github.com/abewley/sort/issues/22
* https://arxiv.org/pdf/1602.00763.pdf
* https://github.com/nwojke/deep_sort
* https://arxiv.org/pdf/1703.07402.pdf
* https://github.com/ambakick/Person-Detection-and-Tracking
* https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98
* https://medium.com/@sshleifer/how-to-finetune-tensorflows-object-detection-models-on-kitti-self-driving-dataset-c8fcfe3258e9
* https://gist.github.com/douglasrizzo/c70e186678f126f1b9005ca83d8bd2ce
* https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-2-converting-dataset-to-tfrecord-47f24be9248d
