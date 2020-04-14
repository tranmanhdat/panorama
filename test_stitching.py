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
    panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))
    cv2.imshow("panorama2", panorama2)
    cv2.waitKey(0)
    temp = panorama2
    panorama2 = panorama2 * mask2
    result=panorama1+panorama2

    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    # final_result = result[min_row:max_row, min_col:max_col, :]
    final_result = result[:,:]
    final_result = final_result.astype(np.uint8)
    return final_result, temp

def main(argv1,argv2):
    img2 = cv2.imread(argv2)
    H = np.load("H_matrix.npy")
    # ret, H_invert = cv2.invert(H)
    # print(H.dot(H_invert))
    H_invert = H
    result = np.zeros((480*3, 640*3, 3),)
    for i in range(0,480):
        for j in range(0,640):
            x = (H_invert[0][0]*i + H_invert[0][1]*j + H_invert[0][2]) / (H_invert[2][0]*i+H_invert[2][1]*j+H_invert[2][2])
            y = (H_invert[1][0]*i + H_invert[1][1]*j + H_invert[1][2]) / (H_invert[2][0]*i + H_invert[2][1]*j + H_invert[2][2])
            if x>-50 and y>-300:
	            result[x.astype(int)+50,y.astype(int)+200, :]= img2[i, j, :]
    result = result.astype(np.uint8)

    paronama = cv2.warpPerspective(img2, H,(640*2, 480*2))
    cv2.imshow('cv2', paronama)
    cv2.imwrite('cv2.jpg', paronama)
    cv2.imshow('resutl', result)
    cv2.imwrite('mul_matrix.jpg',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

H = np.load("H_matrix.npy", allow_pickle=True)


main('image_left.png', 'image_right.png')


