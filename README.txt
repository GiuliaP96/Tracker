OpenCV - cv2
conda install -c menpo opencv
TrackerNew.py
R - press 1 time to stop tracker. 2nd time to restart tracker from the last bounding box before tracking failure

S - start new trial -reset values and ask to select start position

E - end trial and tracking, save log files  

spacebar - pause the video

Simple_Object_Detection.py
Simple tracker - only cnn but missing tensor flow and deepsort 

Object_tracker.py script
Run following command in conda shell
python object_tracker.py --video ./data/video/Rat1faildetect.mp4 --output ./data/video/results.avi --weights ./weights/yolov3-custom.tf --num_classes 3 --classes ./data/labels/obj.names

If missing modules follow README.md in yolov3_deepsort folder
Load_weights.py to convert weights in .tf for tensor flow model 