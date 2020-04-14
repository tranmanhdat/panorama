import cv2
import numpy as np

path_to_image1 = "image_left.png"
path_to_save_image1 = "_image_left.png"
path_to_save_keypoints1 = "keypoint_1"

path_to_image2 = "image_right.png"
path_to_save_image2 = "_image_right.png"
path_to_save_keypoints2 = "keypoint_2"

# path_to_image1 = "image_left.JPG"
# path_to_save_image1 = "_image_left.png"
# path_to_save_keypoints1 = "keypoint_1"
#
# path_to_image2 = "image_right.JPG"
# path_to_save_image2 = "_image_right.png"
# path_to_save_keypoints2 = "keypoint_2"

image1 = cv2.imread(path_to_image1)
image2 = cv2.imread(path_to_image2)

(h1, w1) = image1.shape[:2]
(h2, w2) = image2.shape[:2]
# height = 900
height = h1
r1 = height / float(h1)
r2 = height / float(h2)
# print(r1)
# print(r2)
dim1 = (int(w1 * r1), height)
dim2 = (int(w2 * r2), height)
resized1 = cv2.resize(image1, dim1, interpolation=cv2.INTER_AREA)
resized2 = cv2.resize(image2, dim2, interpolation=cv2.INTER_AREA)
# resized1 = image1
# resized2 = image2
cv2.imshow("image1", resized1)
cv2.imshow("image2", resized2)

list_clicked_point1 = []
point_id1 = 0
list_clicked_point2 = []
point_id2 = 0


def draw_circle1(event, x, y, flags, param, color = (0,0,255)):
    global point_id1
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(resized1, (x, y), 5, color, -1)
        cv2.putText(resized1, str(point_id1), (x + 5, y-5), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        cv2.imwrite(path_to_save_image1, resized1)
        point_id1 += 1
        print(x / r1, y / r1)
        clicked_point = np.array([x /r1, y /r1])
        list_clicked_point1.append(clicked_point)

def draw_circle2(event, x, y, flags, param, color = (0,0,255)):
    global point_id2
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(resized2, (x, y), 5, color, -1)
        cv2.putText(resized2, str(point_id2), (x + 5, y-5), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        cv2.imwrite(path_to_save_image2, resized2)
        point_id2 += 1
        print(x / r2, y / r2)
        clicked_point = np.array([x /r2, y /r2])
        list_clicked_point2.append(clicked_point)


cv2.setMouseCallback('image1', draw_circle1)
cv2.setMouseCallback('image2', draw_circle2)

while True:
    cv2.imshow('image1', resized1)
    cv2.imshow('image2', resized2)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        list_clicked_point1 = np.asarray(list_clicked_point1)
        np.save(path_to_save_keypoints1, list_clicked_point1)
        list_clicked_point2 = np.asarray(list_clicked_point2)
        np.save(path_to_save_keypoints2, list_clicked_point2)
        break
    else:
        continue
