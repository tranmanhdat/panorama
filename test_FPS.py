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
    cv2.imshow('panorama2',panorama2)
    panorama2 = panorama2*mask2
    result=panorama1+panorama2

    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    # print(max_row, max_col)
    # final_result = result[:,:]
    final_result = final_result.astype(np.uint8)
    return final_result


def main(argv1,argv2):
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    H = np.load("H_matrix.npy")
        # if ret1==True and ret2 == True:
        #     cv2.imshow('frame1', img1)
        #     cv2.imshow('frame2', img2)

    final=blending(img1, img2, H)
    # final= cv2.resize(final,(int(final.shape[1]), int(final.shape[0])))
    cv2.imwrite("panorama.png", final)
    cv2.imshow('panorama', final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    try:
        main('image_left.png', 'image_right.png')
    except IndexError:
        print ("Please input two source images: ")
        print ("For example: python Image_Stitching.py '/Users/linrl3/Desktop/picture/p1.jpg' '/Users/linrl3/Desktop/picture/p2.jpg'")
