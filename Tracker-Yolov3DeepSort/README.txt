If new train and weights file [saved in weights folder], to convert weights in TensorFlow format run 
> python load_weights.py --weights ./weights/yolov3.weights --output ./weights/yolov3-custom.tf --num_classes 3 

In conda/anaconda shell run 
#Tensorflow CPU
> conda env create -f conda-cpu.yml
> conda activate tracker-cpu

# Tensorflow GPU
> conda env create -f conda-gpu.yml
> conda activate tracker-gpu

# TensorFlow CPU
> pip install -r requirements.txt

# TensorFlow GPU
> pip install -r requirements-gpu.txt

Check 
> python check_GPU.py

Run Tracker
> cd Single-Multiple-Custom-Object-Detection-and-Tracking
> python object_tracker.py


Net infos

Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input (InputLayer)              [(None, None, None,  0
__________________________________________________________________________________________________
yolo_darknet (Model)            ((None, None, None,  40620640    input[0][0]
__________________________________________________________________________________________________
yolo_conv_0 (Model)             (None, None, None, 5 11024384    yolo_darknet[1][2]
__________________________________________________________________________________________________
yolo_conv_1 (Model)             (None, None, None, 2 2957312     yolo_conv_0[1][0]
                                                                 yolo_darknet[1][1]
__________________________________________________________________________________________________
yolo_conv_2 (Model)             (None, None, None, 1 741376      yolo_conv_1[1][0]
                                                                 yolo_darknet[1][0]
__________________________________________________________________________________________________
yolo_output_0 (Model)           (None, None, None, 3 4747288     yolo_conv_0[1][0]
__________________________________________________________________________________________________
yolo_output_1 (Model)           (None, None, None, 3 1194008     yolo_conv_1[1][0]
__________________________________________________________________________________________________
yolo_output_2 (Model)           (None, None, None, 3 302104      yolo_conv_2[1][0]
__________________________________________________________________________________________________
yolo_boxes_0 (Lambda)           ((None, None, None,  0           yolo_output_0[1][0]
__________________________________________________________________________________________________
yolo_boxes_1 (Lambda)           ((None, None, None,  0           yolo_output_1[1][0]
__________________________________________________________________________________________________
yolo_boxes_2 (Lambda)           ((None, None, None,  0           yolo_output_2[1][0]
__________________________________________________________________________________________________
yolo_nms (Lambda)               ((None, 100, 4), (No 0           yolo_boxes_0[0][0]
                                                                 yolo_boxes_0[0][1]
                                                                 yolo_boxes_0[0][2]
                                                                 yolo_boxes_1[0][0]
                                                                 yolo_boxes_1[0][1]
                                                                 yolo_boxes_1[0][2]
                                                                 yolo_boxes_2[0][0]
                                                                 yolo_boxes_2[0][1]
                                                                 yolo_boxes_2[0][2]
==================================================================================================
Total params: 61,587,112
Trainable params: 61,534,504
Non-trainable params: 52,608