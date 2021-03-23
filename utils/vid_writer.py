import cv2
import numpy as np 


class Writer:
    
    def __init__(self):
        
        self.writer = None
        self._h = None
        self._w = None
        self.output = None
		
        self.codec = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')

    def write(self, frame, saved_to):		
        
        if self.writer is None:
            self._h = frame.shape[0]
            self._w = frame.shape[1]
            self.writer = cv2.VideoWriter(str(saved_to), self.codec, 20, (self._w, self._h), True)
            
        self.output = np.zeros((self._h,self._w,3), dtype = 'uint8')
        self.output[0:self._h, 0:self._w] = frame
        self.writer.write(self.output)
			

