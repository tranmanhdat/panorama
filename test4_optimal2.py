import collections
from threading import Thread
import cv2
import time
import numpy as np
from configparser import ConfigParser
import itertools
(width, height) = 640, 480

class VideoScreenshot(object):
	def __init__(self, src=0, src2=0):
		# Create a VideoCapture object
		self.capture = cv2.VideoCapture(src)
		self.capture2 = cv2.VideoCapture(src2)

		self.config = self.init_params('config.ini')
		self.smoothing_window_size = self.config.getint('args',
		                                                'smoothing_window_size')
		# Take screenshot every x seconds
		self.screenshot_interval = 1

		# Default resolutions of the frame are obtained (system dependent)
		self.frame_width = width
		self.frame_height = height
		self.size = self.frame_width, self.frame_height

		self.H = np.load("H_matrix.npy", allow_pickle=True)
		self.mask_left = self.create_mask('left_image')
		self.mask_right = self.create_mask('right_image')

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
		# out = cv2.VideoWriter('output_blend.avi',
		#                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
		#                       (1900, 960))
		# Display frames in main program
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
				cv2.imshow('result', final)
			# Press Q on keyboard to stop recording
			key = cv2.waitKey(1)
			if key == ord('q'):
				self.capture.release()
				self.capture2.release()
				cv2.destroyAllWindows()
				exit(1)

	def create_mask(self, version):
		smoothing_window_size = self.smoothing_window_size
		height_panorama = self.frame_height
		mask = np.zeros((height_panorama, smoothing_window_size))
		if version == 'left_image':
			mask[:, :] = np.tile(
					np.linspace(1, 0, smoothing_window_size).T,
					(height_panorama, 1))
		else:
			mask[:, :] = np.tile(
					np.linspace(0, 1, smoothing_window_size).T,
					(height_panorama, 1))
		return cv2.merge([mask, mask, mask])

	def blending(self):
		height_img1 = self.frame_height
		width_img1 = self.frame_width
		width_img2 = self.frame_width
		height_panorama = height_img1
		width_panorama = width_img1 + width_img2

		result = np.zeros((height_panorama, width_panorama, 3))

		result[:, 0:self.frame_width - self.smoothing_window_size, :] \
			= self.frame[:, 0:self.frame_width - self.smoothing_window_size, :]
		merger_left = self.frame[:,self.frame_width - self.smoothing_window_size:self.frame_width,:] * self.mask_left

		panorama2 = cv2.warpPerspective(self.frame2, self.H,
		                                (width_panorama, height_panorama))
		cv2.imshow('warpPerspective',panorama2)
		result[:, self.frame_width:, :] = panorama2[:, self.frame_width:, :]
		merger_right = panorama2[:,self.frame_width - self.smoothing_window_size:self.frame_width,:] * self.mask_right

		result[:,self.frame_width - self.smoothing_window_size:self.frame_width,:] = merger_left + merger_right

		min_row, max_row = 0, 480
		min_col, max_col = 0, 900

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
	rtsp_stream_link = 'your stream link!'

	video_stream_widget = VideoScreenshot(
			"rtsp://192.168.1.10:554/user=fitmta_password=fitmta_channel=1_stream=0.sdp?real_stream",
			"rtsp://fitmta:fitmta@192.168.1.120:554/av0_0")
	# video_stream_widget.thread.join()
	# video_stream_widget.thread2.join()
	# while True:
	# try:
	video_stream_widget.show_frame()
# video_stream_widget.stitch_image()
