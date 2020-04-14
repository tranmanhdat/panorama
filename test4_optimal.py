import collections
from threading import Thread
import cv2
import time
import numpy as np
from configparser import ConfigParser
import itertools

class VideoScreenshot(object):
	def __init__(self, src=0, src2=0):
		self.config = self.init_params('config.ini')
		# Create a VideoCapture object
		self.capture = cv2.VideoCapture(src)
		self.capture2 = cv2.VideoCapture(src2)

		# Take screenshot every x seconds
		self.screenshot_interval = 1

		# Default resolutions of the frame are obtained (system dependent)
		self.frame_width = int(self.capture.get(3))
		self.frame_height = int(self.capture.get(4))

		self.H = np.load("H_matrix.npy", allow_pickle=True)

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
		self.fps = 0

	def init_params(self, path_to_ini_file):
		config = ConfigParser()
		config.read(path_to_ini_file)
		return config
	def update(self):
		# Read the next frame from the stream in a different thread
		while True:
			if self.capture.isOpened():
				(self.status, self.frame) = self.capture.read()

	def update2(self):
		# Read the next frame from the stream in a different thread
		while True:
			if self.capture2.isOpened():
				(self.status2, self.frame2) = self.capture2.read()

	def show_frame(self):
		# ID = 0
		# Display frames in main program
		out = cv2.VideoWriter('output_noblend.avi',
		                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
		                      (1900, 960))
		while True:
			if self.status and self.status2:
				start = time.time()
				final = self.blending()
				end = time.time()
				self.fps = self.fps * 0.9 + 1 / (end - start) * 0.1
				fps = "{:.2f}".format(self.fps)
				final = cv2.putText(final, fps, (50, 50),
				                    cv2.FONT_HERSHEY_SIMPLEX,
				                    1, (255, 0, 0), 2, cv2.LINE_AA)
				# out.write(final)
				small_img = cv2.resize(final, (self.frame_width*2//4*3, self.frame_height//4*3))
				cv2.imshow('result', small_img)
				# out.write(final)

			# Press Q on keyboard to stop recording
			key = cv2.waitKey(1)
			if key == ord('q'):
				self.capture.release()
				self.capture2.release()
				cv2.destroyAllWindows()
				exit(1)

	def blending(self):
		height_img1 = self.frame_height
		width_img1 = self.frame_width
		width_img2 = self.frame_width
		height_panorama = height_img1
		width_panorama = width_img1 + width_img2

		result = np.zeros((height_panorama, width_panorama, 3))

		result[0:self.frame_height, 0:self.frame_width, :] = self.frame

		panorama2 = cv2.warpPerspective(self.frame2, self.H,
		                                (width_panorama, height_panorama))
		# result = panorama1 + panorama2
		result[0:self.frame_height, self.frame_width:, :] = panorama2[0:self.frame_height, self.frame_width:, :]

		min_row, max_row = 0, 960
		min_col, max_col = 0, 1900
		final_result = result[min_row:max_row, min_col:max_col, :]
		# final_result = trim(result)
		# final_result = result
		final_result = final_result.astype(np.uint8)
		return final_result

def trim(frame):
	# crop top
	if not np.sum(frame[0]):
		return trim(frame[1:])
	# crop bottom
	elif not np.sum(frame[-1]):
		return trim(frame[:-2])
	# crop left
	elif not np.sum(frame[:, 0]):
		return trim(frame[:, 1:])
	# crop right
	elif not np.sum(frame[:, -1]):
		return trim(frame[:, :-2])
	return frame


if __name__ == '__main__':
	video_stream_widget = VideoScreenshot(
			"rtsp://khanh29bk:Admin123@192.168.0.11/Src/MediaInput/h264/stream_1/ch_",
			"rtsp://khanh29bk:Admin123@192.168.0.12/Src/MediaInput/h264/stream_1/ch_")
	video_stream_widget.show_frame()