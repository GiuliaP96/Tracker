# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:52:20 2021

@author: User
"""

'''
Title: Tracker

Description: A simple offline tracker for detection of rats 
			 in the novel Hex-Maze experiment. Serves as a 
			 replacement for the manual location scorer

Organistaion: Genzel Lab, Donders Institute	
			  Radboud University, Nijmegen

Author(s): Atharva Kand
'''

from utils import mask, kalman_filter
from itertools import groupby
from datetime import date 
from pathlib import Path 
from collections import deque

import cv2
import math
import time
import logging
import argparse
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
        self.frame = None
        self.frame_rate = 0
        self.disp_frame = None
        self.total_detections = -1
        self.trialnum = 1

        self.pos_centroid = None
        self.kf_coords = None
        self.centroid_list = deque(maxlen = 500)          #change maxlen value to chnage how long the pink line is
        self.node_pos = []
        self.node_id = []
        self.saved_nodes = []
        self.KF_age = 0
        
        self.record_detections = False
 
        self.hex_mask = mask.create_mask(self.node_list)
        self.KF = kalman_filter.KF()

        self.run_vid()

    #process and display video 
    def run_vid(self):
        '''
        Frame by Frame looping of video
        '''
        save_flag = 0
        print('loading tracker...\n')
        time.sleep(2.0)
        Rat= None 
        Init=False
        boxes =[]
        tracker=None
    

        with open(self.save, 'a+') as file:
        	file.write(f"Rat number: {self.rat} , Date: {self.date} \n")

        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    self.paused = True

            self.frame_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
            
            #process and display frame
            if self.frame is not None:
                self.disp_frame = self.frame.copy()
                self.disp_frame = cv2.resize(self.disp_frame, (1176, 712)) 
                self.preprocessing(self.disp_frame, Rat, tracker,Init,boxes)
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
                else:
                    logger.info('Rat not detected')

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
            	print('#Program ended by user')
            	break

            elif key == ord('s'):
                self.record_detections = not self.record_detections
                Init=False
                Rat=None
                Rat = cv2.selectROI("Select ROI",self.disp_frame, fromCenter=False,showCrosshair=True)
                save_flag += 1
                tracker =cv2.TrackerCSRT_create()
                
                #condition to save/log data to file upon second press of 's' key
                if save_flag / 2 == 1:
                	self.save_to_file(self.save)
                	self.saved_nodes = []
                	self.node_pos = []
                	self.centroid_list = []
                	self.trialnum += 1
                	save_flag = 0
                else:
                	logger.info('Recording Trial {}'.format(self.trialnum))                
                
                
            elif key == ord('r'):
                self.paused =not self.paused
                Init=False
                Rat=None
                Rat = cv2.selectROI("Select another ROI",self.disp_frame, fromCenter=False,showCrosshair=True)
                save_flag += 1
                self.paused =not self.paused
                tracker =cv2.TrackerCSRT_create()
                if save_flag >= 1:
                	self.save_to_file(self.save)
                	self.saved_nodes = []
                	self.node_pos = []
                	self.centroid_list = []
                	self.trialnum += 1
                	save_flag = 0
                else:
                	logger.info('Recording after failed detection {}'.format(self.trialnum))                


            elif key == ord(' '):
                self.paused = not self.paused

        self.cap.release()
        cv2.destroyAllWindows()

    def preprocessing(self, frame,Rat, tracker, Init, boxes):
        '''
        pre-process frame - apply mask, bg subtractor and morphology operations

        Input: Frame (i.e image to be preprocessed)
        '''
        frame  = np.array(frame)
        i=0
        #apply mask on frame from mask.py 								
        for i in range(0,3):
            frame[:, :, i] = frame[:, :, i] * self.hex_mask		
        #background subtraction and morphology
        backsub = BG_SUB.apply(frame)
        if Rat is not None: ##Selected Roi
                # Initialize tracker with first frame and bounding box if was not init before
                       if(not Init):
                            tracker.init(backsub, Rat)
                            [x0, y0, x1, y1] = Rat      #save first bounding box
                            myBox = (x0, y0, x1, y1)
                            Init = True
                       (found,Rat) = tracker.update(backsub)    #update tracker with new frame bounding box  
                     # self.paused= False

                       counter=0 ##count number of time rat is not found
                       if not found: # Tracking failure
                             counter += 1 #update counter
                             cv2.putText(self.disp_frame, "Tracking failure", (80,200), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,250),1)
                             cv2.putText(self.disp_frame, "Press R to select new ROI", (25,300), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,250),1)
                    #         cv2.putText(self.disp_frame, "Press S to save new trial", (25,340), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,250),1)
                             cv2.putText(self.disp_frame, "Time : " + str(int(self.frame_time)), (50,350), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,250), 1)
                             Rat=boxes[len(boxes)-1] #assigne last boxes for the next frames 
                             if counter>3:
                                 print('counter',counter)
                             ##keep and draw last found box
                             (x, y, w, h) = [int(v) for v in Rat]
                             cv2.rectangle(self.disp_frame, (x, y), (x + w, y + h),( 0, 0,255), 2)
                             
                       if found:# Tracking success 
                                  
                                (x, y, w, h) = [int(v) for v in Rat]
                                myBox = (x, y, w, h)
                                boxes.append(myBox)
                                cv2.rectangle(self.disp_frame, (x, y), (x + w, y + h),(255, 0, 0), 2) ##draw blue bounding box                                 
                                                       
                                black = np.zeros((backsub.shape[0], backsub.shape[1], 3), np.uint8) #---black frame
                                # frame with black everywhere and blank in buonding box position
                                black1 = cv2.rectangle(black,(x,y),(x + w, y + h),(255, 255, 255), -1)   
                                gray = cv2.cvtColor(black1,cv2.COLOR_BGR2GRAY)#---converting to gray
                                #creating mask with ROI
                                ret, mask = cv2.threshold(gray,127,255, 0)
                                ##background subtracked frame masked out of ROI position
                                fin = cv2.bitwise_and(backsub,backsub,mask = mask)#
                                #kernel white
                                kernel = np.ones((5,5),np.uint8)  
                                #morphology operations
                                gradient = cv2.morphologyEx(fin, cv2.MORPH_GRADIENT, kernel)
                                closing = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)
                      #         cv2.imshow('Closing',closing)
                     
                                self.find_contours(closing)
                      
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

            if area > 1:            #MIN_RAT_SIZE = 5 previously
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
                	if points_dist(self.pos_centroid, self.centroid_list[-2]) > 1.5:
                		self.kf_coords = self.KF.estimate()
                		self.pos_centroid = self.centroid_list[-2]


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
        nodes_dict = mask.create_node_dict(self.node_list)				#dictionary of node names and corresponding coordinates
        record = self.record_detections and not self.paused             #condition to go into 'save mode'


        #if the centroid position of rat is within 20 pixels of any node
        #register that node to a list. 
        if self.pos_centroid is not None:
            for node_name in nodes_dict:
                if points_dist(self.pos_centroid, nodes_dict[node_name]) < 20:
                    if record: 
                        self.saved_nodes.append(node_name)						
                        self.node_pos.append(nodes_dict[node_name])
        
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
            cv2.putText(frame, str(self.frame_time), (1110,690), 
                        fontFace = FONT, fontScale = 0.75, color = (0,0,255), thickness = 1)	

            #draw the path that the rat has traversed
            if len(self.centroid_list) >= 2:
                for i in range(1, len(self.centroid_list)):
                    cv2.line(frame, self.centroid_list[i], self.centroid_list[i - 1], 
                             color = (255, 0, 255), thickness = 2)

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
            file.write('\n')



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


    


    
            
        