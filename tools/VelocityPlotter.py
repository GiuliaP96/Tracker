# -*- coding: utf-8 -*-
# flags for after active and inactive
from pathlib import Path
from tkinter import *
from tkinter import Tk, filedialog, Label
import mask
from matplotlib import pyplot as plt
import matplotlib
#matplotlib.use("Qt5Agg")
from matplotlib import cm
from matplotlib.pyplot import figure
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
import csv
import networkx as nx
import numpy as np
import cv2
import statistics as stat
import seaborn as sns
import itertools

def fopen(ftype):
    if ftype == 'nodes':
     root.filename = filedialog.askopenfilename(title = "Select file Log", filetypes = (("All Files", "*.*"), ("Log Files", "*.log")))
    if ftype == 'summary':
     root.filename = filedialog.askopenfilename(title = "Select summary file", filetypes = (("All Files", "*.*"), ("TxtFiles", "*.txt")))
    return root.filename 


def log_reader(fname):  ##summary
    flag = False
    num = None
    savelist = []
    count = 0
  #  palette = itertools.cycle(sns.color_palette())
    name = str(fname)    
    with open(name) as file:
        frame =  np.zeros(shape=[712 , 1176, 3], dtype = np.uint8)
        lines = file.readlines() 
       #viridis = cm.get_cmap('viridis', 256)
        velocity=[]
        timestart=0
        count = 0
        color=0
        node=0
        trials={}
        
        points = []
        for line in lines:
            if 'Recording Trial' in line:
                flag = True                
                points=[]
                count += 1
              #  color +=5
                print('count',count)       
            if flag:
             #   foundtrial(frame,palette,line,flag) 
                  if "(" in line:
                    coords = line.partition("(")[2].partition(")")[0].split(',')
                    points.append((int(coords[0]), int(coords[1])))
                    #count += 1 
            if len(points) >= 2:
                for i in range(1, len(points)): #points[i]
                        append_value(trials=trials, key=count, value=points[i])
    i=0  
    for key in trials:
      print(key) 
      i+=1       
      for x in trials[key]: 
          print('trials',key,trials[key])  
         # print(color[i]) 
     #     plt.plot(x[0],x[1], color=cmap(trialnum / float(20)))
          cv2.circle(frame,x, 1, color =(250,250,250), thickness = -1)#(255, 255, color)
  #  plt.show()
    cv2.imshow('trial path', frame)

      # save frame as JPEG file
   # name = "frame%d.jpg"%trial
   # cv2.imwrite('Frame All Trials',frame)       


def velocities(fname2,nl):   
    name2 = str(fname2)
    flagsum=False
    trials={}
    cmap = mpl.cm.autumn
    st={}  ###space and time
    values=[]
    timespace=[]
    vel=None
    j=0
    count=0
   # nodes_dict = mask.create_node_dict(nl)
                ##read velocities 
    with open(name2,'r') as summary:
           velocities=[]
           
           nx, nsteps = 100, 20
           x = np.linspace(0, 1, nx)
           lines = summary.readlines()
           for line in lines:    
             if 'Summary' in line:  # and flagsum
                   #i+=1                   
                   if len(velocities) >= 2:
                      j+=1                        
                      for i in range(1, len(velocities)): #points[i]
                        append_value(trials=trials, key=j, value=velocities[i])
                        append_value(trials=st, key=j, value=timespace[i])
                      velocities=[]
                      timespace=[]
                   flagsum= False       
             if 'Start' in line:
                 flagsum=True 
                 count=0             
             if flagsum:
                     if (".") in line: #in line
                        values = line.split()
                        timespace.append((float(values[4]),float(values[5])))
                        velocities.append(float(values[-1]))
    summary.close() 
    graph_speed_averages(trials)
   # fig1.show()
    plt.show()
    graph_space_time(st)
    plt.show()
    #fig2.show()
    
 #   graph(st,trials)   

def graph_speed_averages(trials):
    plt.rcParams["figure.figsize"] = (18.5,10.5)
 
    #cmap = mpl.cm.autumn
    
    velocity=[]
    trial_num=[]
    means=[]
    print('SUMMARY AVERAGE VELOCITY')
    for k in trials:
        for vel in trials[k]:
            velocity.append(vel)            ##appen vel                            
        means.append(stat.mean(velocity))   ##calculate mean within trial
        trial_num.append(k)          ##append trial num  
        print('\nTrial number',k,'Mean Velocity:', means[-1])
    plt.plot(trial_num, means,linestyle='--', marker='o', color='k')
   # fig= plt.gcf()
    find_intercept(x=trial_num, y=means)
 #   plt.show()
   # fig1= plt.gcf()
    #fig1.show()
   # fig1.show()
    
def graph_space_time(st): 
    plt.rcParams["figure.figsize"] = (18.5,10.5)
    i=1
    time=[]
    space=[]
    x=[]
    y=[]
      #  print('st', st)
    ##plot space and time for all nodes passed in all trials
    for key in st:        
      for i in range(0,len(st[key])):
          t=round(float(st[key][i][0]),2) 
          s=round(float(st[key][i][1]),2)
          time.append(t)         
          space.append(s)
         
          if i+1 ==len(st[key]):         
           x=np.cumsum(time)
           y=np.cumsum(space)
          
      space=[]
      time=[]              
      #print('trial',key,i,'len',len(st[key]),'time',x,'space',y)  
      
      plot_velocities(x,y,key)
    plt.title('Space/Time all nodes in all Trials',fontsize=24)
    plt.xlabel('Space (meters)',fontsize=20)
    plt.ylabel('Time (seconds)', fontsize=20)              #break  
    plt.legend(loc='right', frameon=False) 
    plt.figure(figsize=((18.5,10.5)))
    plt.show()
    
def plot_velocities(x,y,key):  
 
    #cmap = mpl.cm.autumn
    #for j in range(0,len(x)):
    plt.plot(x,y,label='Trial{}'.format(key),linestyle='--', marker='o')
    
          
def find_intercept(x,y):  
    # Find the slope and intercept of the best fit line        
 slope, intercept = np.polyfit(x, y, 1)

# Create a list of values in the best fit line
 abline_values = [slope * i + intercept for i in x]

# Plot the best fit line over the actual values
 plt.plot(x, y, '--')
 plt.plot(x, abline_values, 'r')
 plt.text(9,0.3, s='slope:{}'.format(round(slope,6)), fontsize=20)
 plt.title('Mean Velocity all Trials',fontsize=24) 
 plt.xlabel('Trial number',fontsize=20)
 plt.ylabel('Speed (m/s)',fontsize=20) 
 plt.figure(figsize=((18.5,10.5)))
 plt.show()

         
def append_value(trials, key, value):
    # Check if key exist in dict or not
    if key in trials:
        # Key exist in dict.
        # Check if type of value of key is list or not
        if not isinstance(trials[key], list):
            # If type is not list then make it list
            trials[key] = [trials[key]]
        # Append the value in list
        trials[key].append(value)
    else:
        # As key is not in dict,
        # so, add key-value pair
        trials[key] = value
    return trials
        
        
def foundtrial(frame,palette, line, flag):  
     points = []
     if "(" in line:
        coords = line.partition("(")[2].partition(")")[0].split(',')
        points.append((int(coords[0]), int(coords[1])))
                    #count += 1 
        if len(points) >= 2:
           for i in range(1, len(points)):
                cv2.circle(frame, points[i], 1, color = next(palette), thickness = -1)#(255, 255, color)
                if 'Recording' in line:
                            flag=False
                            points=[]
                            return flag
        cv2.imshow('trial path', frame)
                        
             
if __name__ == "__main__":
#    from os.path import abspath
    root = Tk()
    root.withdraw()
    root.title('Converter')

    filename1 = fopen(ftype = 'nodes')
    log_path1 = Path(filename1).resolve()
    filename2 = fopen(ftype = 'summary')
    log_path2 = Path(filename2).absolute()
   # log_reader(fname=filename1) 
    
    node_list = Path('node_list_new.csv').resolve()
    nodelist = 'new_list.csv'
    nodes_dict = mask.create_node_dict(nodelist)
    nl = str(node_list)
    velocities(filename2,nl)
 