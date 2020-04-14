import collections
from threading import Thread
import cv2
import time
import numpy as np
import itertools
from queue import Queue


class VideoScreenshot(object):
	def __init__(self, src=0, src2=0, queue_size=64):
		# Create a VideoCapture object
		self.capture = cv2.VideoCapture(src)
		self.capture2 = cv2.VideoCapture(src2)

		# Create 2 Queue for store frames read from streams video
		self.Q1 = Queue(maxsize=queue_size)
		self.Q2 = Queue(maxsize=queue_size)
		# Take screenshot every x seconds
		self.screenshot_interval = 1
		self.stop = False

		# Default resolutions of the frame are obtained (system dependent)
		self.frame_width = int(self.capture.get(3))
		self.frame_height = int(self.capture.get(4))
		print(self.frame_width, self.frame_height)

		self.H = np.load("H_matrix.npy", allow_pickle=True)
		self.mask_left = self.create_mask('left_image')
		self.mask_right = self.create_mask('right_image')

		# Start the thread to read frames from the video stream
		self.thread = Thread(target=self.update1, args=())
		self.thread.daemon = True
		self.thread.start()
		time.sleep(0.2)

		self.thread2 = Thread(target=self.update2, args=())
		self.thread2.daemon = True
		self.thread2.start()
		time.sleep(0.2)
		self.status = False
		self.status2 = False
		self.frame = None
		self.frame2 = None
		# self.result = None
		self.fps = 0

	def update1(self):
		# Read the next frame from the stream in a different thread
		while True:
			if self.stop:
				return
			if not self.Q1.full():
				(self.status, frame) = self.capture.read()
				self.Q1.put(frame)

	def update2(self):
		# Read the next frame from the stream in a different thread
		while True:
			if self.stop:
				return
			if not self.Q2.full():
				(self.status2, frame) = self.capture2.read()
				self.Q2.put(frame)
	def show_frame(self):
		# ID = 0
		# Display frames in main program
		while not self.stop:
			print(self.Q1.qsize(), self.Q2.qsize())
			start = time.time()
			final = self.blending()
			end = time.time()
			self.fps = self.fps * 0.9 + 1 / (end - start) * 0.1
			fps = "{:.2f}".format(self.fps)
			final = cv2.putText(final, fps, (50, 50),
			                    cv2.FONT_HERSHEY_SIMPLEX,
			                    1, (255, 0, 0), 2, cv2.LINE_AA)
			cv2.imshow('result', final)
			# cv2.imwrite('temp/' + str(ID) + '.jpg', final)
			# ID = ID + 1
			# cv2.imwrite('temp/paranoma.jpg', final)
			# cv2.imshow("frame1", self.frame)
			# cv2.imshow("frame2", self.frame2)

			# Press Q on keyboard to stop recording
			key = cv2.waitKey(1)
			if key == ord('q'):
				self.capture.release()
				self.capture2.release()
				cv2.destroyAllWindows()
				exit(1)

	def stitch_image(self):
		if self.frame is not None and self.frame2 is not None:
			start = time.time()
			final = self.blending()
			end = time.time()
			self.fps = self.fps * 0.8 + 1 / (end - start) * 0.2
			fps = "{:.2f}".format(self.fps)
			# print("here3")
			# cv2.imwrite('panorama.png', final)
			self.result = cv2.putText(self.result, fps, (50, 50),
			                          cv2.FONT_HERSHEY_SIMPLEX,
			                          1, (255, 0, 0), 2, cv2.LINE_AA)
			cv2.imshow('panorama', self.result)
			key = cv2.waitKey(1)
			if key == ord('q'):
				self.capture.release()
				self.capture2.release()
				cv2.destroyAllWindows()
				exit(1)

	# def stitch_image(self):
	#     print("here2")
	#     final = blending(self.frame1, self.frame2, self.H)
	#     cv2.imwrite('panorama_{}.png'.format(self.frame_count), final)
	#     self.frame_count += 1
	#     cv2.imshow('panorama', final)
	#     cv2.waitKey(1)

	def create_mask(self, version):
		smoothing_window_size = 800
		height_img1 = self.frame_height
		width_img1 = self.frame_width
		width_img2 = self.frame_width
		height_panorama = height_img1
		width_panorama = width_img1 + width_img2
		offset = int(smoothing_window_size / 2)
		barrier = self.frame_width - int(smoothing_window_size / 2)
		mask = np.zeros((height_panorama, width_panorama))
		if version == 'left_image':
			mask[:, barrier - offset:barrier + offset] = np.tile(
					np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
			# print('first : ', mask)
			mask[:, :barrier - offset] = 1
		# print('second ', mask)
		else:
			mask[:, barrier - offset:barrier + offset] = np.tile(
					np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
			mask[:, barrier + offset:] = 1
		return cv2.merge([mask, mask, mask])

	def blending(self):
		frame1 = self.Q1.get()
		frame2 = self.Q2.get()
		height_img1 = self.frame_height
		width_img1 = self.frame_width
		width_img2 = self.frame_width
		height_panorama = height_img1
		width_panorama = width_img1 + width_img2

		panorama1 = np.zeros((height_panorama, width_panorama, 3))
		mask1 = self.mask_left
		panorama1[0:self.frame_height, 0:self.frame_width, :] = frame1
		panorama1 *= mask1
		mask2 = self.mask_right
		panorama2 = cv2.warpPerspective(frame2, self.H,
		                                (width_panorama, height_panorama)) *mask2
		# print(type(panorama2))
		result = panorama1 + panorama2

		#using np.where
		# rows, cols = np.where(result[:, :, 0] != 0)
		# min_row, max_row = min(rows), max(rows) + 1
		# min_col, max_col = min(cols), max(cols) + 1

		# min_row, max_row = 0, 960
		# min_col, max_col = 0, 1382

		# using for-loop
		# height , width, _ = result.shape
		# min_row, max_row, min_col, max_col = height+1, 0, width+1, 0
		# for idx, val in enumerate(result[:,:,0].ravel()):
		# 	if val != 0:
		# 		result_div = divmod(idx, width)
		# 		row = int(result_div[0])
		# 		col = int(result_div[1])
		# 		if row<min_row:
		# 			min_row = row
		# 		if row> max_row:
		# 			max_row = row
		# 		if col<min_col:
		# 			min_col = col
		# 		if col>max_col:
		# 			max_col = col
		# final_result = result[min_row:max_row, min_col:max_col, :]
		final_result = trim(result)
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
			"/media/trandat/Entertainment/Video/1.mp4",
			"/media/trandat/Entertainment/Video/2.mp4")

	# H = np.load("H_matrix.npy")
	# video_stream_widget.thread.join()
	# video_stream_widget.thread2.join()
	# while True:
	# try:
	video_stream_widget.show_frame()
# video_stream_widget.stitch_image()
