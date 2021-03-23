import numpy as np
import cv2

class KF():

	def __init__(self):
		self.kf = cv2.KalmanFilter(4, 2)

		self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
										[0, 1, 0, 0]], dtype = np.float32)
		self.kf.transitionMatrix = np.array([[1, 0 , 1, 0],
										[0, 1, 0, 1],
										[0, 0, 1, 0],
										[0, 0, 0, 1]], dtype = np.float32)

	def estimate(self):
		predict  = self.kf.predict()
		return predict

	def correct(self, x_loc, y_loc):
		measurement = np.array([[np.float32(x_loc)], [np.float32(y_loc)]])
		self.kf.correct(measurement)

