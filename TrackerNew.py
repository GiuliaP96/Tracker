# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:52:20 2021


"""

'''
Title: Tracker

Description: A simple offline tracker for detection of rats 
             in the novel Hex-Maze experiment. Serves as a 
             replacement for the manual location scorer

Organistaion: Genzel Lab, Donders Institute    
              Radboud University, Nijmegen

Author(s): Atharva Kand-Giulia Porro
'''

from itertools import groupby
from datetime import date, timedelta, datetime
from pathlib import Path 
from collections import deque
from utils import mask, kalman_filter

import cv2
import math
import time
import logging
#import argparse
import os
import numpy as np

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
BG_SUB = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 150, detectShadows = False)

FONT = cv2.FONT_HERSHEY_TRIPLEX
RT_FPS = 25

MIN_RAT_SIZE = 5



#find the shortest distance between two points in space
def points_dist(p1, p2):
    dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return dist

#convert time in milli seconds to -> hh:mm:ss,uuu format
def convert_milli(time):
    sec = (time / 1000) % 60
    minute = (time / (1000*60)) % 60
    hr = (time / (1000*60*60)) % 24

    return f'{int(hr):02d}:{int(minute):02d}:{sec:.3f}'


class Tracker:
    def __init__(self, vp, nl, file_id):
        '''Tracker class initialisations'''
        nsp = str(date.today()) + '_' + file_id
        self.save = os.path.join(gui.save_path, nsp) + '{}'.format('.txt')
        self.node_list = str(nl)
        self.cap = cv2.VideoCapture(str(vp))

        #experiment meta-data
        self.rat = input("Enter rat number: ")
        self.date = input("Enter date of trial: ")

        self.paused = False
        self.failed = False
        self.frame = None
        self.frame_rate = 0
        self.frame_count= None
        self.disp_frame = None
        self.total_detections = -1
        self.trialnum = 0

        self.pos_centroid = None
        self.kf_coords = None
        self.centroid_list = deque(maxlen = 500)         #change maxlen value to chnage how long the pink line is
        self.node_pos = []
        self.time_points= []
        self.node_id = []
        self.saved_nodes = []
        self.saved_velocities=[]
        self.time_points=[]  ##EXTRA
        self.summary_trial=[]
        self.KF_age = 0
        self.Rat=None
        self.tracker=None
        self.Init=None
        self.myBox=None
        self.fail_flag= 0
        
        self.record_detections = False
 
        self.hex_mask = mask.create_mask(self.node_list)
        self.KF = kalman_filter.KF()

        self.run_vid()

    #process and display video 
    def run_vid(self):
        '''
        Frame by Frame looping of video
        '''
        print('loading tracker...\n')
        time.sleep(2.0)
        
    
        with open(self.save, 'a+') as file:
            file.write(f"Rat number: {self.rat} , Date: {self.date} \n")

        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    self.paused = True

            self.frame_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count=  self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.converted_time = convert_milli(int(self.frame_time))
            
            #process and display frame
            if self.frame is not None:
                self.disp_frame = self.frame.copy()
                self.disp_frame = cv2.resize(self.disp_frame, (1176, 712)) 
                self.preprocessing(self.disp_frame)  #, Rat, tracker,Init,boxes
                self.annotate_frame(self.disp_frame)
                cv2.imshow('Tracker', self.disp_frame)    

            #log present centroid position if program is in 'save mode'
            if self.record_detections and not self.paused:
                if self.pos_centroid is not None:
                    converted_time = convert_milli(int(self.frame_time))
                    if self.saved_nodes:
                        logger.info(f'{converted_time} : The rat position is: {self.pos_centroid} @ {self.saved_nodes[-1]}')
                  
                    else:
                        logger.info(f'{converted_time} : The rat position is: {self.pos_centroid}')
                
               # else:
                   # logger.info('Rat not detected')

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                #condition to save/log data to file upon press of 's' key creating new trial in log file
                self.failed= False
                self.record_detections = not self.record_detections
                self.saved_nodes = []
                self.node_pos = []
                self.centroid_list = []
                self.trialnum += 1
                logger.info('Recording Trial {}'.format(self.trialnum)) 
                
                self.Init=False
                self.time_points=[]
                self.summary_trial=[]
                self.Rat = cv2.selectROI("Select ROI",self.disp_frame, fromCenter=False,showCrosshair=True)
                self.tracker =cv2.TrackerCSRT_create()
                

            elif key == ord('r'):
                ##re-initialize tracker with previous found bounding box, same trialnum
               # self.record_detections = not self.record_detections
               self.tracker=None
               self.fail_flag +=1
               logger.info('Failed detection {}'.format(self.fail_flag))
               if self.fail_flag/2==1:
                self.Init=False
                self.Rat=self.myBox
                self.fail_flag=0
                self.tracker =cv2.TrackerCSRT_create()  
                              
                    
            elif key == ord('e'):
                ##condition to save Trailnum to file and calculate velocity
                self.tracker=None
                self.record_detections = not self.record_detections
                self.calculate_velocity(self.time_points)
                self.save_to_file(self.save)                

            elif key == ord(' '):
                self.paused = not self.paused
                
            elif key == ord('q'):
                print('#Program ended by user')
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def preprocessing(self, frame): #,Rat, tracker, Init, boxes
        '''
        pre-process frame - apply mask, bg subtractor and morphology operations

        Input: Frame (i.e image to be preprocessed)
        '''
        if self.tracker is not None:
            frame  = np.array(frame)
        #apply mask on frame from mask.py                                 
            for i in range(0,3):
                 frame[:, :, i] = frame[:, :, i] * self.hex_mask        
            #background subtraction and morphology 
            backsub = BG_SUB.apply(frame)
             # cv2.imshow('frame tracker',backsub)
       
         ##Selected Roi
                # Initialize tracker with first frame and bounding box if was not init before
            if not self.Init:
                            self.tracker.init(backsub, self.Rat)
                           # [x0, y0, x1, y1] = self.Rat      #save first bounding box                            
                            self.Init = True
                        #update tracker in next frame (found=float,self.Rat= x,y,width,height)
                     
            (found,self.Rat) = self.tracker.update(backsub)    #update tracker with new frame bounding box                  
                       
            if not found and self.frame is not None: # Tracking failure
                             self.failed=True
                            # logger.info('Failed detection {}'.format(fail_flag)) 
                             cv2.putText(self.disp_frame, "Tracking failure", (90,170), cv2.FONT_HERSHEY_TRIPLEX, 0.80,(0,0,250),2)
                            
                             ##keep and draw last roi found 
                             (x, y, w, h) = [int(v) for v in self.myBox]
                             cv2.rectangle(self.disp_frame, (x, y), (x + w, y + h),( 0, 0,255),2) 
                             
                             ##keep tracking contours in previous roi
                             black = np.zeros((backsub.shape[0], backsub.shape[1], 3), np.uint8)
                             black_ROI = cv2.rectangle(black,(x,y),(x + w, y + h),(255, 255, 255), -1)   
                             gray = cv2.cvtColor(black_ROI,cv2.COLOR_BGR2GRAY)
                             ret, mask = cv2.threshold(gray,127,255, 0)
                             masked_ROI = cv2.bitwise_and(backsub,backsub,mask = mask)
                             self.find_contours(masked_ROI)
                            
                            
            if found:# Tracking success 
                                (x, y, w, h) = [int(v) for v in self.Rat]
                                self.myBox =  x, y, w, h #self.Rat                                 
                                cv2.rectangle(self.disp_frame, (x, y), (x + w, y + h),(0,255, 0), 1) ##draw blue bounding box                                 
                                
                                ##morphology operation to filter out pixels not inside bounding box 
                                black = np.zeros((backsub.shape[0], backsub.shape[1], 3), np.uint8) #---black frame
                                # frame with black everywhere and blank in buonding box position
                                black_ROI = cv2.rectangle(black,(x,y),(x + w, y + h),(255, 255, 255), -1)   
                                gray = cv2.cvtColor(black_ROI,cv2.COLOR_BGR2GRAY)#---converting to gray
                                #creating mask with ROI
                                ret, mask = cv2.threshold(gray,127,255, 0)
                                
                                ##background subtracked frame masked out of ROI position
                                masked_ROI = cv2.bitwise_and(backsub,backsub,mask = mask)#
                               # cv2.imshow('Macked roi',masked_ROI)
                                self.find_contours(masked_ROI)
                      
    def find_contours(self, frame):
        '''
        Function to find contours

        '''
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)          #_, cont, _ in new opencv version
        detections_in_frame = 0
       
        #create lists of centroid x and y means 
        cx_mean= []             
        cy_mean = []
        
        #find contours greater than the minimum area and 
        #caluclate means of the of all such contours 
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >MIN_RAT_SIZE:            #prev  = 5  5
            #    cv2.drawContours(self.disp_frame, contour, -1, (0,255,0), 3)
                contour_moments = cv2.moments(contour)           
                cx = int(contour_moments['m10'] / contour_moments['m00'])
                cy = int(contour_moments['m01'] / contour_moments['m00'])
                cx_mean.append(cx)
                cy_mean.append(cy)
                detections_in_frame += 1
                self.total_detections += 1
            else:
                continue
                
        #find centroid position by calculating means of x and y of contour centroids
        #if the program is on 'save mode', add centroid poisitions to list. if no 
        #detections are in frame, assume rat is stationary, i.e centroid = previous 
        #centroid. if distance between prev centroid and centroid > 2 pixels , 
        #centroid = previous centroid  
        if self.total_detections:
                if detections_in_frame != 0:
                    self.pos_centroid = (int(sum(cx_mean) / len(cx_mean)), 
                                        int(sum(cy_mean) / len(cy_mean)))
                    if self.record_detections:
                        self.centroid_list.append(self.pos_centroid)
                    #cv2.circle(frame, self.pos_centroid, 20, color = (0, 69, 255), thickness = 1)                    
                else:
                    if self.record_detections and self.centroid_list:
                        self.pos_centroid = self.centroid_list[-1]
                    #cv2.circle(frame, self.pos_centroid, 20, color = (0, 69, 255), thickness = 1)

                if len(self.centroid_list) > 2:
                    if points_dist(self.pos_centroid, self.centroid_list[-2]) > 2:
                        self.kf_coords = self.KF.estimate()
                        self.pos_centroid = self.centroid_list[-2]
                        

    def calculate_velocity(self,time_points): #
    
    ##calculate rat speed between two consecutive nodes 
    ##result e.g.summary trial - [ [('209', '210'), (16.76, 17.56), 0.375] , [('210', '211'), (17.56, 17.88), 0.937],]  
 
      bridges = { ('124', '201'):0.60,
           ('121', '302'):1.72,
           ('223', '404'):1.69,
           ('324', '401'):0.60,
           ('305', '220'):0.60}
      if len(time_points) > 3:
            lenght=0
            self.first_node= time_points[0][1]            
            format = '%H:%M:%S.%f' 
            # first_time=((time_points[i][0])/ 1000) % 60 
            
        ##iterate over list of touple with time points and nodes IDs
        ###grab start time and node name and next node         
            for i in range(0, len(time_points)):
              start_node= time_points[i][1]
              start_time= datetime.strptime((time_points[i][0]), format).time()
              j=i+1
              if j == len(time_points):
                self.last_node= time_points[i][1]                
              else:
                end_node= time_points[j][1]
                end_time=datetime.strptime((time_points[j][0]), format).time()
                difference = timedelta(hours= end_time.hour-start_time.hour, minutes= end_time.minute-start_time.minute, seconds=end_time.second-start_time.second, microseconds=end_time.microsecond-start_time.microsecond).total_seconds()
                if (start_node, end_node) in bridges:
                          lenght= bridges[(start_node, end_node)]
                          
                elif(end_node, start_node) in bridges:
                        lenght= bridges[(end_node, start_node)] 
                        
                else:
                          lenght=0.30    ##30cm within islands
                speed= round(lenght/difference, 3)
                self.summary_trial.append([(start_node,end_node),(time_points[i][0],time_points[j][0]),difference,lenght,speed])
                self.saved_velocities.append(speed)

            
    @staticmethod
    def annotate_node(frame, point, node):
        '''Annotate traversed nodes on to the frame

        Input: Frame (to be annotated), Point: x, y coords of node, Node: Node name
        '''
        cv2.circle(frame, point, 20, color = (0, 69, 255), thickness = 1)
        cv2.putText(frame, str(node), (point[0] + 2, point[1] + 2), 
                    fontScale=0.5, fontFace=FONT, color = (0, 69, 255), thickness=1,
                    lineType=cv2.LINE_AA)


    def annotate_frame(self, frame):
        '''
        Annotates frame with frame information, path and nodes resgistered

        '''
        nodes_dict = mask.create_node_dict(self.node_list)                #dictionary of node names and corresponding coordinates
        record = self.record_detections and not self.paused             #condition to go into 'save mode'
        #fail_detection = self.record_detections and not self.paused and self.failed
        cv2.putText(frame, str(self.converted_time), (970,670), 
                        fontFace = FONT, fontScale = 0.75, color = (240,240,240), thickness = 1)
        #if the centroid position of rat is within 20 pixels of any node
        #register that node to a list. 
        if self.pos_centroid is not None:
            for node_name in nodes_dict:
                if points_dist(self.pos_centroid, nodes_dict[node_name]) < 20:
                    if record: 
                        self.saved_nodes.append(node_name)                        
                        self.node_pos.append(nodes_dict[node_name])
                        
                        ###save timepoints for speed calculation
                        if len(self.time_points) <= 0:  
                           self.time_points.append([self.converted_time,node_name])
                        if node_name != self.saved_nodes[(len(self.saved_nodes))-2]:
                               self.time_points.append([self.converted_time,node_name])
         
      #  self.calculate_velocity(self.time_points)

        #annotate all nodes the rat has traversed
        for i in range(0, len(self.saved_nodes)):
            self.annotate_node(frame, point = self.node_pos[i], node = self.saved_nodes[i])


        #frame annotations during recording
        if record:
            
            #savepath  = self.vid_save_path + '{}'.format('.mp4')
            cv2.putText(frame,'Trial:' + str(self.trialnum), (60,60), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            cv2.putText(frame,'Currently writing to file...', (60,80), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            
            cv2.putText(frame, "R: stop/restart tracker", (45,225), cv2.FONT_HERSHEY_TRIPLEX, 0.65,(250,250,250),1)
            cv2.putText(frame, "E: end trial", (45,250), cv2.FONT_HERSHEY_TRIPLEX, 0.65,(250,250,250),1)
            cv2.putText(frame, "S: start new trial", (45,200), cv2.FONT_HERSHEY_TRIPLEX, 0.65,(250,250,250),1)
            cv2.putText(frame, "Frame count : " + str(self.frame_count), (870,640), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0,0,250), 1)

         #   cv2.putText(frame, "FPS : " + str(self.frame_rate), (900,620), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0,0,250), 1)


            #draw the path that the rat has traversed
            if len(self.centroid_list) >= 2:
                for i in range(1, len(self.centroid_list)):
                    cv2.line(frame, self.centroid_list[i], self.centroid_list[i - 1], 
                             color = (255, 0, 255), thickness = 1)

            if self.pos_centroid is not None:
                cv2.line(frame, (self.pos_centroid[0] - 5, self.pos_centroid[1]), (self.pos_centroid[0] + 5, self.pos_centroid[1]), 
                color = (0, 255, 0), thickness = 2)
                cv2.line(frame, (self.pos_centroid[0], self.pos_centroid[1] - 5), (self.pos_centroid[0], self.pos_centroid[1] + 5), 
                color = (0, 255, 0), thickness = 2)
                    
        elif self.paused:
            cv2.putText(frame,'Trial:' + str(self.trialnum), (60,60), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            cv2.putText(frame,'Paused', (60,80), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
                        
    
    #save recorded nodes to file
    def save_to_file(self, fname):
        savelist = []
        with open(fname, 'a+') as file:
            for k, g in groupby(self.saved_nodes):
                savelist.append(k)         
            file.writelines('%s,' % items for items in savelist)
            file.write('\nSummary Trial {}\nStart-Next Nodes// Time points(s) //Seconds//Lenght(cm)// Velocity(m/s)\n'.format(self.trialnum))            
            for i in range(0, len(self.summary_trial)):
                   line=" ".join(map(str,self.summary_trial[i]))
                   file.write(line + '\n')
            file.write('\n')
        file.close()
                

if __name__ == "__main__":
    today  = date.today()
    # parser = argparse.ArgumentParser(description = 'Enter required paths')
    # parser.add_argument('-i', '--id',  type = str, help = 'enter unique file id')
    # args = parser.parse_args()
    
    enter = input('Enter unique file name: ')
    file_id = '' if not enter else enter

    print('#\nLite Tracker version: v1.03\n#\n')
    import utils.gui as gui

    #logger intitialisations
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)

    logfile_name = 'logs/log_{}_{}.log'.format(str(today), file_id)

    fh = logging.FileHandler(str(logfile_name))
    formatter = logging.Formatter('%(levelname)s : %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh) 

    node_list = Path('tools/node_list_new.csv').resolve()
    vid_path = gui.vpath
    logger.info('Video Imported: {}'.format(vid_path))
    print('creating log files...')
    
    Tracker(vp = vid_path, nl = node_list, file_id = file_id)


    


    
            
        