# -*- coding: utf-8 -*-
# @Author: Atharva Kand
# @Date:   2020-08-06 16:21:37
# @Last Modified by:   Atharva Kand

# @Last Modified time: 2020-09-03 14:40:59



import os
import cv2
import math
import numpy as np 
import concurrent.futures

import networkx as nx 


COLOR_SELECTED = (240, 60, 60)
COLOR_UNSELECTED = (255, 255, 255)
IMAGE_SCALER = 1.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 0, 0)




def points_dist(p1, p2):
	'''Calculate distance between two points'''
	dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
	return dist


class Generator:

	def __init__(self, vp, fn):

		self.fname = str(fn) 
		self.cap = cv2.VideoCapture(str(vp))

		self.frame = None
		self.paused = False
		self.selected = False

		self.clicked_point = None
		self.selected_node = None
		self.nodes = {}
		self.points = []

		cv2.namedWindow('Generator', cv2.WINDOW_AUTOSIZE)
		cv2.setMouseCallback('Generator', self.process_input)

		self.mask_im = np.zeros([800, 600, 3], dtype = np.uint8)

		self.loop()

	def loop(self):
		gray = cv2.cvtColor(self.mask_im, cv2.COLOR_BGR2GRAY)
		self.mask_im = np.array(gray)
		
		#loop video frames
		while True:
			if not self.paused:
				ret, self.frame = self.cap.read()
				if not ret:
					self.paused = True

			if self.frame is not None:
				self.annotate_node(self.frame)
				self.frame = cv2.resize(self.frame, (int(600 / IMAGE_SCALER), int(800 / IMAGE_SCALER)))
				cv2.imshow('Generator', self.frame)
				cv2.imshow('mask', self.mask_im)

			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				print('#Program ended by user')
				break

			elif key == ord('s'):
				self.save(self.fname)

			elif key == ord('m'):
				self.create_mask()

			elif key == ord(' '):
				self.paused = not self.paused

		self.cap.release()
		cv2.destroyAllWindows()


	def process_input(self, event, x, y, flag, params):
		'''Handle mouse and keyboard events'''

		#add new node to be saved
		if flag == (cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_LBUTTON) and event == cv2.EVENT_LBUTTONDOWN:
			self.clicked_point = (x , y)
			i = 0
			node_name = ''
			while True:
				cv2.imshow('Generator', self.frame)
				k = cv2.waitKey(0)
				if k == ord('e'):
					break
				else:
					node_name += chr(k)
					cv2.putText(self.frame, chr(k), (x + i + 10, y + 10), fontFace = FONT, fontScale = FONT_SCALE, 
						color = FONT_COLOR, thickness = 2, lineType = cv2.LINE_AA)
					i += 10

			self.nodes[node_name] = self.clicked_point
			self.selected = False

		#select previously added node
		elif event == cv2.EVENT_RBUTTONDOWN and self.nodes:
			self.selected_node = (x , y)
			for k, v in self.nodes.items():
				if points_dist(self.selected_node, v) < 10:
					self.selected = True
					self.selected_node = v
					#self.annotate_node(self.frame, v, COLOR_SELECTED)

		#add unamed location for mask
		elif event == cv2.EVENT_LBUTTONDOWN:
			self.clicked_point = (x , y)
			for k, v in self.nodes.items():
				if points_dist(self.clicked_point, v) < 10:
					self.clicked_point = v
					break

			self.selected = False


	def create_mask(self):
		'''self.mask_im and self.frame image annotations'''
		cv2.line(self.mask_im, (int(self.clicked_point[0] * IMAGE_SCALER), int(self.clicked_point[1] * IMAGE_SCALER))
			, (int(self.selected_node[0] * IMAGE_SCALER), int(self.selected_node[1] * IMAGE_SCALER)), (255, 255, 255), 15)
		cv2.line(self.frame, self.clicked_point, self.selected_node, color = COLOR_SELECTED, thickness = 3)
		cv2.circle(self.frame, self.selected_node, 10, color = COLOR_UNSELECTED, thickness = -1)
		self.clicked_point = None
		self.selected_node = None

	def annotate_node(self, frame):
		'''Annotate nodes to self.frame'''
		if self.selected:
			cv2.circle(frame, self.selected_node, 10, color = COLOR_SELECTED, thickness = -1)
		else:
			if self.clicked_point:
				cv2.circle(frame, self.clicked_point, 10, color = COLOR_UNSELECTED, thickness = -1)

	def save(self, fname):
		'''save mask'''

		#save to .npy file -> loadout with np.load()
		npfile  = os.path.join(gui.save_path, fname)
		self.mask = self.mask_im > 250
		np.save(npfile, self.mask)
		
		#save node positions to .csv file
		csvfile = os.path.join(gui.save_path, fname + '.csv')
		with open(csvfile, 'w') as nf:
			for nn, nl in self.nodes.items():
				nf.write(f'{nn}, {int(nl[0] * IMAGE_SCALER)}, {int(nl[1] * IMAGE_SCALER)}\n')


if __name__ == "__main__":
    #today  = date.today()
    #parser = argparse.ArgumentParser(description = 'Enter required paths')
    #parser.add_argument('-i', '--id',  type = str, help = 'enter unique file id')
    #args = parser.parse_args()

    import gui

    vid_path = gui.vpath

    file_name = input('Enter file name: ')

    Generator(vid_path, file_name)





