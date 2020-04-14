import collections
from threading import Thread
import cv2
import time
import numpy as np
from configparser import ConfigParser
import itertools

(width, height) = 640, 480

class VideoScreenshot(object):
	def __init__(self, src=0, src2=0, get_image=False):
		self.get_image = get_image
		# Create a VideoCapture object
		self.capture = cv2.VideoCapture(src)
		self.capture2 = cv2.VideoCapture(src2)
		# Default resolutions of the frame are obtained (system dependent)
		self.frame_width = width
		self.frame_height = height
		self.size = self.frame_width, self.frame_height
		# self.frame_width = int(self.capture.get(3))
		# self.frame_height = int(self.capture.get(4))
		# print(self.frame_width, self.frame_height)
		# self.frame_width = int(self.capture2.get(3))
		# self.frame_height = int(self.capture2.get(4))
		# print(self.frame_width, self.frame_height)
		# Start the thread to read frames from the video stream
		self.thread = Thread(target=self.update)
		self.thread.daemon = True
		self.thread.start()

		self.thread2 = Thread(target=self.update2)
		self.thread2.daemon = True
		self.thread2.start()

		self.status = False
		self.status2 = False
		self.frame = None
		self.frame2 = None

	# self.result = None

	def update(self):
		# Read the next frame from the stream in a different thread
		while True:
			if self.capture.isOpened():
				(self.status, frame) = self.capture.read()
				self.frame = cv2.resize(frame, self.size)

	def update2(self):
		# Read the next frame from the stream in a different thread
		while True:
			if self.capture2.isOpened():
				(self.status2, frame2) = self.capture2.read()
				self.frame2 = cv2.resize(frame2, self.size)

	def show_frame(self):
		# ID = 0

		# Display frames in main program
		while True:
			if self.status and self.status2:
				# if self.get_image:
				# 	# gray1 = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
				# 	# gray2 = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2GRAY)
				# 	# pattern_size = (4, 6)
				# 	# ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size,
				# 	#                                            None)
				# 	# ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size,
				# 	#                                            None)
				# 	# img1 = cv2.putText(gray1, str(ret1), (50, 50),
				# 	#                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
				# 	#                    cv2.LINE_AA)
				# 	# img2 = cv2.putText(gray2, str(ret2), (50, 50),
				# 	#                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
				# 	#                    cv2.LINE_AA)
				# 	cv2.imshow('img1', img1)
				# 	cv2.imshow('img2', img2)
				# 	if ret1 and ret2:
				# 		cv2.imwrite('image_left.png', self.frame)
				# 		cv2.imwrite('image_right.png', self.frame2)
				# 		self.capture.release()
				# 		self.capture2.release()
				# 		cv2.destroyAllWindows()
				# 		exit(1)
				# else:
				cv2.imshow('left', self.frame)
				cv2.imshow('right', self.frame2)

				# Press Q on keyboard to stop recording
				key = cv2.waitKey(1)
				if key == ord('q'):
					cv2.imwrite('image_left.png', self.frame)
					cv2.imwrite('image_right.png', self.frame2)
					self.capture.release()
					self.capture2.release()
					cv2.destroyAllWindows()
					exit(1)


if __name__ == '__main__':
	rtsp_stream_link = 'your stream link!'

	# video_stream_widget = VideoScreenshot(
	# 		"rtsp://khanh29bk:Admin123@192.168.0.11/Src/MediaInput/h264/stream_1/ch_",
	# 		"rtsp://khanh29bk:Admin123@192.168.0.12/Src/MediaInput/h264/stream_1/ch_")
	video_stream_widget = VideoScreenshot(
					"rtsp://192.168.1.10:554/user=fitmta_password=fitmta_channel=1_stream=0.sdp?real_stream",
					"rtsp://fitmta:fitmta@192.168.1.120:554/av0_0",
					True)
	time.sleep(1)
	# video_stream_widget.thread.join()
	# video_stream_widget.thread2.join()
	# while True:
	# try:
	video_stream_widget.show_frame()
# video_stream_widget.stitch_image()
