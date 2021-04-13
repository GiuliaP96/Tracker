from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt
 
import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import tools.gui as gui

from pathlib import Path


#class Tracker   
def tracker(vp, nl,file_id ): 
 class_names = [c.strip() for c in open('./data/labels/obj.names').readlines()]
 yolo = YoloV3(classes=len(class_names))
 yolo.load_weights('./weights/yolov3-custom.tf')

 max_cosine_distance = 0.8 
 nn_budget = None
 nms_max_overlap = 1  ##1 if want to print every found box 

 model_filename = 'model_data/mars-small128.pb'
 encoder = gdet.create_box_encoder(model_filename, batch_size=1)
 metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
 tracker = Tracker(metric)
 

 vid = cv2.VideoCapture(str(vp)) #'./data/video/Rat1faildetect.mp4'
 codec = cv2.VideoWriter_fourcc(*'MP4V')
 #codec = cv2.VideoWriter_fourcc(*'XVID') .avi
 out_name = file_id+ '.mp4' ##output video name
 vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
 vid_width,vid_height =  int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
 #nsp = str(date.today()) + '_' + file_id
 #save = os.path.join(gui.save_path, nsp) + '{}'.format('.txt')
 out = cv2.VideoWriter('./Results/{}'.format(out_name), codec, vid_fps, (vid_width, vid_height))   #.
# print('output')

 from _collections import deque
 pts = [deque(maxlen=50) for _ in range(1000)] #datapoints cannot be larger than 30 in lenght

 counter_human = []
 counter_rat = []

 while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   # cv2.imshow(img_in)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

    t1 = time.time()

    boxes, scores, classes, nums = yolo.predict(img_in)

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    current_countR = int(0)
    current_countH = int(0)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr()  ##get x y w h bounding box
        class_name= track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                    +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2)

        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        
        
        pts[track.track_id].append(center)

        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)
##here create space-maskhaxe maze
        height, width, _ = img.shape
        #cv2.line(img, (0, int(3*height/6+height/20)), (width, int(3*height/6+height/20)), (0, 255, 0), thickness=2)
        #cv2.line(img, (0, int(3*height/6-height/20)), (width, int(3*height/6-height/20)), (0, 255, 0), thickness=2)
        
        #cv2.line(img, (0, int(height-10+height/20)), (width, int(3*height/6+height/20)), (0, 255, 0), thickness=2)
       # cv2.line(img, (0, int(3*height/6-height/20)), (width, int(3*height/6-height/20)), (0, 255, 0), thickness=2)
           
        center_y = int(((bbox[1])+(bbox[3]))/2)
        if class_name == 'researcher':
                counter_human.append(int(track.track_id))
                current_countH += 1
        
        if class_name == 'rat' or class_name == 'head':
                counter_rat.append(int(track.track_id))
                current_countR += 1
                
                ##count object only if between line draw above - hex maze 
      #  if center_y <= int(3*height/6+height/20) and center_y >= int(3*height/6-height/20):
        #    if class_name == 'rat' or class_name == 'researcher':
         #       counter.append(int(track.track_id))
         #       current_count += 1
    
    total_count_rat = len(set(counter_rat))
    cv2.putText(img, "Current Rat Count: " + str(total_count_rat), (25,130), 0, 1, (250,250,250), 2)
    total_count_human = len(set(counter_human))
    cv2.putText(img, "Current Human Count: " + str(total_count_human), (30,160), 0, 1, (250,250,250), 2)

    fps = 1./(time.time()-t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
    out.write(img)
    img=cv2.resize(img,(1176, 712))
    #cv2.resizeWindow('output', (1176, 712))
    cv2.imshow('output frame', img)
    

    if cv2.waitKey(1) == ord('q'):
        break
 vid.release()
 out.release()
 cv2.destroyAllWindows()

if __name__ == "__main__":
   # today  = date.today()
    # parser = argparse.ArgumentParser(description = 'Enter required paths')
    # parser.add_argument('-i', '--id',  type = str, help = 'enter unique file id')
    # args = parser.parse_args()
    
    enter = input('Enter name output video: ')
    file_id = '' if not enter else enter

    print('#\nLite Tracker version: v1.04\n#\n')
    

   # logger intitialisations
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)

   # logfile_name = 'logs/log_{}_{}.log'.format(str(today), file_id)

   # fh = logging.FileHandler(str(logfile_name))
    #formatter = logging.Formatter('%(levelname)s : %(message)s')
 #   fh.setFormatter(formatter)
  #
    #logger.addHandler(fh) 

    node_list = Path('tools/node_list_new.csv').resolve()
    vid_path =   gui.vpath #'./Results'
    logger.info('Video Imported: {}'.format(vid_path))
    print('creating log files...')
    
    tracker(vp = vid_path, nl = node_list, file_id = file_id)