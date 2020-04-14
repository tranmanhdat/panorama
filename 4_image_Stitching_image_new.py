import cv2
import numpy as np


def create_mask(img1,img2,version):
    smoothing_window_size = 60
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


def main(argv1,argv2):
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    H = np.load("H_matrix.npy")

    src_point = (250, 400)
    src_point_np = np.concatenate([np.array(src_point), np.array([1])])
    src_point_np = np.array([src_point_np]).T
    color = (0, 0, 255)

    # draw point in right hand side
    cv2.circle(img2, src_point, 5, color, -1)
    cv2.imshow("right", img2)

    dst_point_np = np.matmul(H, src_point_np)
    # dst_point_np = cv2.perspectiveTransform(np.array([np.array([np.array(src_point)])]).astype(
    #     np.float32), H)
    # dst_point_np = cv2.warpPerspective(
    #         np.array([np.array([np.array(src_point)])]).astype(
    #                     np.float32), H, (1000,1000))
    print(dst_point_np)
    dst_point_np[0][0] = dst_point_np[0][0]/dst_point_np[2][0]
    dst_point_np[1][0] = dst_point_np[1][0]/dst_point_np[2][0]
    print(dst_point_np.shape)
    # cv2.imshow('dst', dst_point_np)
    # cv2.waitKey(0)
    dst_point = (int(dst_point_np[0][0]), int(dst_point_np[1][0]))
    # print(dst_point)

    # draw point in left hand side
    cv2.circle(img1, dst_point, 5, color, -1)
    cv2.imshow("left", img1)


    final=blending(img1, img2, H)
    cv2.imwrite("panorama.png", final)
    # cv2.imshow('panorama', final)
    cv2.waitKey(0)


if __name__ == '__main__':
    main("image_left.png", "image_right.png")

