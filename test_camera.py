import collections
from threading import Thread
import cv2
import time
import numpy as np
from configparser import ConfigParser
import itertools


class VideoScreenshot(object):
	def __init__(self, src=0):
		# Create a VideoCapture object
		self.capture = cv2.VideoCapture(src)
		# Default resolutions of the frame are obtained (system dependent)
		self.frame_width = int(self.capture.get(3))
		self.frame_height = int(self.capture.get(4))
		print(self.frame_width, self.frame_height)

		# Start the thread to read frames from the video stream
		self.thread = Thread(target=self.update)
		self.thread.daemon = True
		self.thread.start()

		self.status = False
		self.status2 = False
		self.frame = None
		self.frame2 = None

	# self.result = None

	def update(self):
		# Read the next frame from the stream in a different thread
		while True:
			if self.capture.isOpened():
				(self.status, self.frame) = self.capture.read()


	def show_frame(self):
		# ID = 0
		# Display frames in main program
		while True:
			if self.status and self.status2:
				# cv2.imshow('result', final)
				cv2.imshow('left', self.frame)
			# Press Q on keyboard to stop recording
			key = cv2.waitKey(1)
			if key == ord('q'):
				cv2.imwrite('image_left.png', self.frame)
				self.capture.release()
				cv2.destroyAllWindows()
				exit(1)


if __name__ == '__main__':
	# rtsp_stream_link = 'your stream link!'
	#
	# video_stream_widget = VideoScreenshot(
	# 		"rtsp://admin:D9799B@192.168.1.10/554")
	# # video_stream_widget.thread.join()
	# # video_stream_widget.thread2.join()
	# # while True:
	# # try:
	# video_stream_widget.show_frame()
# video_stream_widget.stitch_image()
	cap = cv2.VideoCapture('rtsp://fitmta:fitmta@192.168.1.120:554/av0_0')
# 	cap = cv2.VideoCapture('rtsp://192.168.1.10:554/user=fitmta_password=fitmta_channel=1_stream=0.sdp?real_stream')

	while True:

		# print('About to start the Read command')
		ret, frame = cap.read()
		frame = cv2.resize(frame, (640,460))
		# print('About to show frame of Video.')
		print(frame.shape)

		cv2.imshow("Capturing", frame)
		# print('Running..')

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()