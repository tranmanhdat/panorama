import cv2
import numpy as np

# width, height = 640, 480
#
# if __name__ == '__main__':
#     out1 = cv2.VideoWriter('output1.avi',
#                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
#                           (width, height))
#     out2 = cv2.VideoWriter('output2.avi',
#                            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
#                            (width, height))
#     capture = cv2.VideoCapture('output_concat.avi')
#     while True:
#         ret, frame = capture.read()
#         if ret:
#             out1.write(frame[:,0:640,:])
#             out2.write(frame[:, 640:, :])
#             # out2.write(right)
#         else:
#             out1.release()
#             out2.release()
#             capture.release()
#             break

import cv2
import numpy as np


def create_mask(version):
    height_panorama = frame_height
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


frame_height = 480
frame_width = 640
smoothing_window_size = 60
mask_left = create_mask('left_image')
mask_right = create_mask('right_image')
width = 850
# blen
def blending(image_left, image_right, H):
    height_img1 = frame_height
    width_img1 = frame_width
    width_img2 = frame_width
    height_panorama = height_img1
    width_panorama = width_img1 + width_img2

    result = np.zeros((height_panorama, width_panorama, 3))

    result[:, 0:frame_width - smoothing_window_size, :] \
        = image_left[:, 0:frame_width - smoothing_window_size, :]
    merger_left = image_left[:,
                  frame_width - smoothing_window_size:frame_width,
                  :] * mask_left

    panorama2 = cv2.warpPerspective(image_right, H,
                                    (width_panorama, height_panorama))
    result[:, frame_width:, :] = panorama2[:, frame_width:, :]
    merger_right = panorama2[:,
                   frame_width - smoothing_window_size:frame_width,
                   :] * mask_right

    result[:, frame_width - smoothing_window_size:frame_width,
    :] = merger_left + merger_right

    min_row, max_row = 0, 480
    min_col, max_col = 0, width

    final_result = result[min_row:max_row, min_col:max_col, :]
    # final_result = trim(result)
    # final_result = result
    final_result = final_result.astype(np.uint8)
    return final_result

#no blend
def no_blending(image_left, image_right, H):
    height_img1 = frame_height
    width_img1 = frame_width
    width_img2 = frame_width
    height_panorama = height_img1
    width_panorama = width_img1 + width_img2

    result = np.zeros((height_panorama, width_panorama, 3))

    result[0:frame_height, 0:frame_width, :] = image_left

    panorama2 = cv2.warpPerspective(image_right, H,
                                    (width_panorama, height_panorama))
    # result = panorama1 + panorama2
    result[0:frame_height, frame_width:, :] = panorama2[
                                                        0:frame_height,
                                                        frame_width:, :]

    min_row, max_row = 0, 480
    min_col, max_col = 0, width
    final_result = result[min_row:max_row, min_col:max_col, :]
    # final_result = trim(result)
    # final_result = result
    final_result = final_result.astype(np.uint8)
    return final_result

def main(argv1, argv2):
    cap1 = cv2.VideoCapture(argv1)
    cap2 = cv2.VideoCapture(argv2)
    H = np.load("H_matrix.npy")
    out = cv2.VideoWriter('output_no_blend_850x480.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          (width, 480))
    # out = cv2.VideoWriter('output_blend_850x480.avi',
    #                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
    #                       (width, 480))
    while True:
        ret1, image_left = cap1.read()
        ret2, image_right = cap2.read()
        if ret1 and ret2:
            final = no_blending(image_left, image_right, H)
            # final = blending(image_left, image_right, H)
            cv2.imshow('panorama', final)
            out.write(final)
            cv2.waitKey(1)
        else:
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            out.release()
            break


if __name__ == '__main__':
    main('output_left.avi', 'output_right.avi')
