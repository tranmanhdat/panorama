from threading import Thread
import cv2
import time
import numpy as np


def create_mask(img1,img2,version):
    smoothing_window_size = 800
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 +width_img2
    offset = int(smoothing_window_size / 2)
    barrier = img1.shape[1] - int(smoothing_window_size / 2)
    mask = np.zeros((height_panorama, width_panorama))
    if version== 'left_image':
        mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
        mask[:, :barrier - offset] = 1
    else:
        mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
        mask[:, barrier + offset:] = 1
    return cv2.merge([mask, mask, mask])


def blending(img1,img2, H):
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 + width_img2

    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    mask1 = create_mask(img1,img2,version='left_image')
    panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
    panorama1 *= mask1
    mask2 = create_mask(img1,img2,version='right_image')
    panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
    result=panorama1+panorama2

    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    return final_result


class VideoScreenshot(object):
    def __init__(self, src=0, windows_name=""):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # Take screenshot every x seconds
        self.screenshot_interval = 1

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        # init windows name
        self.windows_name = windows_name

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self):
        # Display frames in main program
        if self.status:
            cv2.imshow(self.windows_name, self.frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def save_frame(self):
        # Save obtained frame periodically
        self.frame_count = 0
        def save_frame_thread():
            while True:
                try:
                    cv2.imwrite('frame_{}.png'.format(self.frame_count), self.frame)
                    self.frame_count += 1
                    time.sleep(self.screenshot_interval)
                except AttributeError:
                    pass
        Thread(target=save_frame_thread, args=()).start()


if __name__ == '__main__':
    rtsp_stream_link = 'your stream link!'
    video_stream_widget = VideoScreenshot("rtsp://khanh29bk:Admin123@192.168.0.11/Src/MediaInput/h264/stream_1/ch_", "camera1")
    # video_stream_widget.save_frame()

    # video_stream_widget2 = VideoScreenshot("rtsp://khanh29bk:Admin123@192.168.0.12/Src/MediaInput/h264/stream_1/ch_", "camera2")
    # video_stream_widget2.save_frame()

    # H = np.load("H_matrix.npy")

    while True:
        try:
            video_stream_widget.show_frame()
            # video_stream_widget2.show_frame()
            # frame1 = video_stream_widget.frame
            # frame2 = video_stream_widget2.frame
            # print("here")
            # final = blending(frame1, frame2, H)
            # cv2.imshow("panorama", final)
            # cv2.write("panorama.jpg", final)
            print("saved panorama")
        except AttributeError:
            pass