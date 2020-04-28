import time
from configparser import ConfigParser
from threading import Thread

import cv2
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression

(width, height) = 640, 480


def cv_draw_boxes(detections, img):
    number_box = 0
    if len(detections)==0:
        return img, 0
    for detection in detections:
        xA, yA, xB, yB = detection[0], detection[1], detection[2], detection[3]
        pt1 = (xA, yA)
        pt2 = (xB, yB)
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
        number_box = number_box + 1
    return img,number_box

class VideoScreenshot(object):
    def __init__(self, src=0, src2=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)
        self.capture2 = cv2.VideoCapture(src2)

        self.config = self.init_params('config.ini')
        self.smoothing_window_size = self.config.getint('args',
                                                        'smoothing_window_size')
        self.write_video = self.config.getboolean('args', 'write_video')
        self.step_frame = self.config.getint('detect', 'step_frame')
        self.thresh = self.config.getfloat('detect', 'thresh')
        self.config_path = self.config.get('detect', 'config_path')
        self.weight_path = self.config.get('detect', 'weight_path')
        self.meta_path = self.config.get('detect', 'meta_path')
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

        self.thread_blend = None
        self.thread_detect = None

        self.status = False
        self.status2 = False
        self.frame = None
        self.frame2 = None
        self.frame_blend = None
        self.frame_detect = None
        # self.result = None
        self.number_boxes = 0
        self.frame_count = 0
        self.fps_blend = 0
        self.fps_detect = 0
        # for detect
        self.hog = None

    def init_params(self, path_to_ini_file):
        config = ConfigParser()
        config.read(path_to_ini_file)
        return config

    def init_detector(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    def create_mask(self, version):
        height_panorama = self.frame_height
        mask = np.zeros((height_panorama, self.smoothing_window_size))
        if version == 'left_image':
            mask[:, :] = np.tile(
                    np.linspace(1, 0, self.smoothing_window_size).T,
                    (height_panorama, 1))
        else:
            mask[:, :] = np.tile(
                    np.linspace(0, 1, self.smoothing_window_size).T,
                    (height_panorama, 1))
        return cv2.merge([mask, mask, mask])

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

    def blending(self):
        height_img1 = self.frame_height
        width_img1 = self.frame_width
        width_img2 = self.frame_width
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        result = np.zeros((height_panorama, width_panorama, 3))
        result[:, 0:self.frame_width - self.smoothing_window_size, :] \
            = self.frame[:, 0:self.frame_width - self.smoothing_window_size, :]
        merger_left = self.frame[:,
                      self.frame_width - self.smoothing_window_size:self.frame_width,
                      :] * self.mask_left
        panorama2 = cv2.warpPerspective(self.frame2, self.H,
                                        (width_panorama, height_panorama))
        result[:, self.frame_width:, :] = panorama2[:, self.frame_width:, :]
        merger_right = panorama2[:,
                       self.frame_width - self.smoothing_window_size:self.frame_width,
                       :] * self.mask_right
        result[:,
        self.frame_width - self.smoothing_window_size:self.frame_width,
        :] = merger_left + merger_right
        min_row, max_row = 0, self.config.getint('args', 'max_row')
        min_col, max_col = 0, self.config.getint('args', 'max_col')
        final_result = result[min_row:max_row, min_col:max_col, :]
        final_result = final_result.astype(np.uint8)
        return final_result

    def blend(self):
        while True:
            if self.status and self.status2:
                start = time.time()
                self.frame_blend = self.blending()
                end = time.time()
                if self.fps_blend == 0:
                    self.fps_blend = 1 / (end - start) * 0.1
                else:
                    self.fps_blend = self.fps_blend * 0.9 + 1 / (end - start) * 0.1

    def start_blend(self):
        self.thread_blend = Thread(target=self.blend)
        self.thread_blend.daemon = True
        self.thread_blend.start()

    def show_frame(self):
        # w,h = 850, 480
        if self.write_video:
        # ID = 0
            out = cv2.VideoWriter('output_blend.avi',
                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                                  self.size)
        # Display frames in main program
        while True:
            if self.frame_detect is not None:
                # frame = cv2.resize(self.frame_detect, (w*2, h*2))
                frame = imutils.resize(self.frame_detect,
                               width=max(800, self.frame_detect.shape[1]))
                cv2.imshow('detect with hog', frame)
                if self.write_video:
                    out.write(self.frame_detect)
            # Press Q on keyboard to stop recording
            key = cv2.waitKey(1)
            if key == ord('q'):
                if self.write_video:
                    out.release()
                self.capture.release()
                self.capture2.release()
                cv2.destroyAllWindows()
                exit(1)
    def detect(self):
        while True:
            self.frame_count = self.frame_count +1
            detections = []
            # frame_detect = cv2.cvtColor(self.frame_blend, cv2.COLOR_BGR2RGB)
            frame_detect = imutils.resize(self.frame_blend, width=min(400, self.frame_blend.shape[1]))
            start = time.time()
            if self.frame_count % self.step_frame == 0:
                (rects, weights) = self.hog.detectMultiScale(frame_detect, winStride=(4, 4),
                                                        padding=(8, 8),
                                                        scale=1.05)
                for i, box in enumerate(rects):
                    if weights[i]>self.thresh:
                        detections.append(box)
                detections = np.array(
                        [[x, y, x + w, y + h] for (x, y, w, h) in rects])
                detections = non_max_suppression(detections, probs=None,
                                           overlapThresh=0.65)
                self.frame_count = 0
            end = time.time()
            fps_cur = 1/(end-start)
            self.fps_detect = fps_cur if self.fps_detect==0 else (self.fps_detect*0.95+fps_cur*0.05)
            image, box_count = cv_draw_boxes(detections, frame_detect)
            self.number_boxes = self.number_boxes + box_count
            fps_str = "{:.2f}".format(self.fps_detect)
            cv2.putText(image, fps_str , (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            self.frame_detect = image
    def start_detect(self):
        self.thread_detect = Thread(target=self.detect)
        self.thread_detect.daemon = True
        self.thread_detect.start()
if __name__ == '__main__':
    video_stream_widget = VideoScreenshot(
            "rtsp://fitmta:fitmta@192.168.1.120:554/av0_0",
            "rtsp://192.168.1.10:554/user=fitmta_password=fitmta_channel=1_stream=0.sdp?real_stream")
    video_stream_widget.init_detector()
    video_stream_widget.start_blend()
    time.sleep(1)
    video_stream_widget.start_detect()
    video_stream_widget.show_frame()
