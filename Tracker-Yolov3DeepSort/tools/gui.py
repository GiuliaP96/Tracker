# -*- coding: utf-8 -*-

from pathlib import Path
from tkinter import *
from tkinter import Tk, filedialog



def fopen(message, ftype):
	if ftype == 'node':
		root.filename = filedialog.askopenfilename(title = message, 
			filetypes = [("CSV Files", "*.csv"), ("All Files", "*.*")])
	elif ftype == 'video':
		root.filename = filedialog.askopenfilename(title = message, 
			filetypes = [("Video Files", [".mp4", ".avi", ".mkv"]), ("All Files", "*.*")])
	elif ftype == 'dir':
		root.filename = filedialog.askdirectory(title = message, initialdir = '/Destktop')

	return root.filename

if __name__ == '__main__':
	print('Open gui from tracker.py')

else:
	root = Tk()
	root.withdraw()
	#root.title('TrackerGUI')
	
	vfile = fopen('Select Video File', ftype = 'video')
	vpath = Path(vfile).resolve()

	print('video file imported...')

	savedir = fopen('Select Save Directory', ftype = 'dir')
	save_path = Path(savedir).resolve()

	#root.mainloop()

	



