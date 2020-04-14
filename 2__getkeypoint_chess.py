import cv2
import numpy as np

path_to_image1 = "image_left.png"
path_to_save_image1 = "_image_left.png"
path_to_save_keypoints1 = "keypoint_1"

path_to_image2 = "image_right.png"
path_to_save_image2 = "_image_right.png"
path_to_save_keypoints2 = "keypoint_2"


image1 = cv2.imread(path_to_image1)
image2 = cv2.imread(path_to_image2)

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

(h1, w1) = image1.shape[:2]
(h2, w2) = image2.shape[:2]
# height = 900

pattern_size = (4,6)
# resized1 = image1
# resized2 = image2

# cv2.imshow("image1", gray1)
# cv2.imshow("image2", gray2)

# cv2.waitKey(0)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size, None)
ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size, None)

print(ret1, ret2)

corners1 = cv2.cornerSubPix(gray1,corners1, (11,11), (-1,-1), criteria)
corners2 = cv2.cornerSubPix(gray2,corners2, (11,11), (-1,-1), criteria)

img1 = cv2.drawChessboardCorners(image1, pattern_size, corners1,ret1)
img2 = cv2.drawChessboardCorners(image2, pattern_size, corners2,ret2)

img1 = cv2.putText(img1,str(ret1),(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
img2 = cv2.putText(img2,str(ret2),(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey(0)

cv2.imwrite(path_to_save_image1, img1)
cv2.imwrite(path_to_save_image2, image2)

np.save(path_to_save_keypoints1, corners1)
np.save(path_to_save_keypoints2, corners2)