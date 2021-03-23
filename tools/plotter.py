# -*- coding: utf-8 -*-
# flags for after active and inactive
from pathlib import Path
from tkinter import *
from tkinter import Tk, filedialog, Label

import numpy as np
import cv2

def fopen():
    root.filename = filedialog.askopenfilename(title = "Select file", filetypes = (("All Files", "*.*"), ("Log Files", "*.log")))
    return root.filename 
def fopen2():   
    root.filename2 = filedialog.askopenfilename(title = "Select file", filetypes = (("All Files", "*.*"), ("Log Files", "*.log")))
    return root.filename2 

def log_reader(fname):
    flag = False
    num = None
    savelist = []
    count = 0

    name = str(fname)

    with open(name) as file:
        frame =  np.zeros(shape=[712 , 1176, 3], dtype = np.uint8)
        lines = file.readlines()
        points = []
        count = 0 
        
        for line in lines:
            if 'Recording ' in line:
                flag = True
                count += 1
            
            if flag:
                if "(" in line:
                    coords = line.partition("(")[2].partition(")")[0].split(',')
                    points.append((int(coords[0]), int(coords[1])))
                    count += 1 


        if len(points) >= 2:
            for i in range(1, len(points)):
                cv2.circle(frame, points[i], 1, color = (255, 255, 255), thickness = -1)
        
        image = cv2.imread('image.jpg', cv2.IMREAD_UNCHANGED)
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dsize = (width, height)
        dst = cv.addWeighted(frame, alpha, image, beta, gamma[, dst[, dtype]]) 
        
        output = cv2.resize(frame, dsize)
        cv2.imwrite('cv2-resize-image-50.png',output)
        cv2.imshow('trial path', frame)
        #scale_percent = 50
#calculate the 50 percent of original dimensions
#  
#
# dsize
#
# resize image
#
#cv2.imshow('Redimension', output)
# 
        ##plot 2 images
        #cv2.imshow("image 1", my_image_1)
        #cv2.imshow("image 2", my_image_2)
        #cv2.waitKey(0)
        #image = cv2.imread('base.jpg')
        #image = cv2.cvtColor(image.cv2.COLOR_)
        #image2 = cv2.resize(image, (712, 1176, 3))
if __name__ == "__main__":
    
    root = Tk()
    root.title('Converter')
    file = fopen()
    
    log_path = Path(file).resolve()

    log_reader(log_path) 
    my_label = Label(root, text = 'Your file has been converted successfully').pack()
    root.mainloop()
           