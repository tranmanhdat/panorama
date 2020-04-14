import cv2
import numpy as np
import sys

def compute_H():
    image1_kp = np.load("keypoint_1.npy")
    image2_kp = np.load("keypoint_2.npy")
    #revert array
    # image2_kp = np.flip(image2_kp,0)
    H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
    return H

def main():
    H = compute_H()
    print(H)
    np.save("H_matrix", H)


if __name__ == '__main__':
    main()
