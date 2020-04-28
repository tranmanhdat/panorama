import collections
from threading import Thread
import cv2
import time

import numpy as np

image1_kp = np.load("keypoint_1.npy")
image2_kp = np.load("keypoint_2.npy")
image1_kp = image1_kp.astype(int)
image2_kp = image2_kp.astype(int)
(width, height) = 640, 480


class VideoScreenshot(object):
    def __init__(self, src=0, src2=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)
        self.capture2 = cv2.VideoCapture(src2)
        # Take screenshot every x seconds
        self.screenshot_interval = 1
        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = width
        self.frame_height = height
        self.size = self.frame_width, self.frame_height
        self.status1 = False
        self.status2 = False
        self.status = False
        self.frame1 = None
        self.frame2 = None
        self.frame = np.zeros(
                (self.frame_height, self.frame_width * 2, 3)).astype(np.uint8)
        self.fps = 0
        self.H = np.load("H_matrix.npy", allow_pickle=True)
        # Start the thread to read frames from the video stream
        self.thread = None
        self.thread1 = None
        self.thread2 = None

    def open(self):
        self.thread1 = Thread(target=self.update1)
        self.thread1.daemon = True
        self.thread1.start()
        self.thread2 = Thread(target=self.update2)
        self.thread2.daemon = True
        self.thread2.start()
        print('Camera Open!')

    def start(self):
        self.thread = Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()
        print('Camera Started!')

    def update1(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status1, frame) = self.capture.read()
                self.frame1 = cv2.resize(frame, self.size)

    def update2(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture2.isOpened():
                (self.status2, frame2) = self.capture2.read()
                self.frame2 = cv2.resize(frame2, self.size)

    def update(self):
        while True:
            if self.status1 and self.status2:
                # frame1 = self.frame1
                # frame2 = self.frame2
                # frame1 = self.drawROI(frame1, image1_kp)
                # frame2 = self.drawROI(frame2, image2_kp)
                # self.frame[:, 0:self.frame_width, :] = frame1
                # self.frame[:, self.frame_width:, :] = frame2
                self.status = True

    def show_frame(self):
        # ID = 0
        out_l = cv2.VideoWriter('output_left.avi',
                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (self.frame_width, self.frame_height))
        out_r = cv2.VideoWriter('output_right.avi',
                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (self.frame_width, self.frame_height))
        # Display frames in main program
        while True:
            # if self.status1 and self.status2:
            # 	# start = time.time()
            # 	# final = self.blending()
            # 	# end = time.time()
            # 	# self.fps = self.fps * 0.9 + 1 / (end - start) * 0.1
            # 	# fps = "{:.2f}".format(self.fps)
            # 	# final = cv2.putText(final, fps, (50, 50),
            # 	#                     cv2.FONT_HERSHEY_SIMPLEX,
            # 	#                     1, (255, 0, 0), 2, cv2.LINE_AA)
            # 	# # out.write(final)
            # 	# cv2.imshow('result', final)
            if self.status:
                cv2.imshow('Left', self.frame1)
                cv2.imshow('right', self.frame2)
                out_l.write(self.frame1)
                out_r.write(self.frame2)
            # Press Q on keyboard to stop recording
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.capture.release()
                self.capture2.release()
                out_l.release()
                out_r.release()
                cv2.destroyAllWindows()
                exit(1)
    def drawROI(self, img, points):
        for i in range(0, len(points)-1):
            img = cv2.line(img, (points[i][0],points[i][1]), (points[i+1][0],points[i+1][1]), (0, 0, 255), 2)
        img = cv2.line(img, (points[-1][0],points[-1][1]), (points[0][0],points[0][1]), (0, 0, 255), 2 )
        return img
if __name__ == '__main__':
    video_stream_widget = VideoScreenshot(
            "rtsp://fitmta:fitmta@192.168.1.120:554/av0_0",
            "rtsp://192.168.1.10:554/user=fitmta_password=fitmta_channel=1_stream=0.sdp?real_stream")
    video_stream_widget.open()
    time.sleep(1)
    video_stream_widget.start()
    video_stream_widget.show_frame()
