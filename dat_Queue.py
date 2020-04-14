from threading import Thread
import cv2
import time
import numpy as np
from queue import Queue

class VideoScreenshot(object):
    def __init__(self, src=0, src2=0, queue_size=64):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)
        self.capture2 = cv2.VideoCapture(src2)

        #2 queues for storage images
        self.queue_image_1 = Queue(maxsize=queue_size)
        self.queue_image_2 = Queue(maxsize=queue_size)
        # Take screenshot every x seconds
        self.screenshot_interval = 1

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        # self.H = np.load("H_matrix.npy")

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.thread2 = Thread(target=self.update2, args=())
        self.thread2.daemon = True
        self.thread2.start()

        self.status = False
        self.status2 = False
        self.frame = None
        self.frame2 = None

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened() and :
                (self.status, self.frame) = self.capture.read()

    def update2(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture2.isOpened():
                (self.status2, self.frame2) = self.capture2.read()

    def show_frame(self):
        # Display frames in main program
        if self.status and self.status2:
            cv2.imshow("frame1", self.frame)
            cv2.imshow("frame2", self.frame2)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            self.capture2.release()
            cv2.destroyAllWindows()
            exit(1)


    def stitch_image(self):
        if self.frame is not None and self.frame2 is not None:
            final = blending(self.frame, self.frame2, self.H)
            print("here3")
            cv2.imwrite('panorama.png', final)
            cv2.imshow('panorama', final)
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


def create_mask(img1, img2, version):
    smoothing_window_size = 800
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 + width_img2
    offset = int(smoothing_window_size / 2)
    barrier = img1.shape[1] - int(smoothing_window_size / 2)
    mask = np.zeros((height_panorama, width_panorama))
    if version == 'left_image':
        mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
        mask[:, :barrier - offset] = 1
    else:
        mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
        mask[:, barrier + offset:] = 1
    return cv2.merge([mask, mask, mask])


def blending(img1, img2, H):
    print("here1")
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 + width_img2

    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    mask1 = create_mask(img1, img2, version='left_image')
    panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
    panorama1 *= mask1
    mask2 = create_mask(img1, img2, version='right_image')
    panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama)) * mask2
    result = panorama1 + panorama2

    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    return final_result


if __name__ == '__main__':
    rtsp_stream_link = 'your stream link!'

    video_stream_widget = VideoScreenshot("rtsp://khanh29bk:Admin123@192.168.0.11/Src/MediaInput/h264/stream_1/ch_",
                                          "rtsp://khanh29bk:Admin123@192.168.0.12/Src/MediaInput/h264/stream_1/ch_")

    # H = np.load("H_matrix.npy")

    while True:
        # try:
        video_stream_widget.show_frame()
        # video_stream_widget.stitch_image()
