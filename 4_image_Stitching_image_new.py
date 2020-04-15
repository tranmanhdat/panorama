import cv2
import numpy as np
import math


def drawROI(img, points):
    color = (0, 255, 0)
    for i in range(0, len(points) - 1):
        img = cv2.line(img, (points[i][0], points[i][1]),
                       (points[i + 1][0], points[i + 1][1]), color, 2)
    img = cv2.line(img, (points[-1][0], points[-1][1]),
                   (points[0][0], points[0][1]), color, 2)
    return img


height, width = 480, 640
image1_kp = np.load("keypoint_1.npy")
image2_kp = np.load("keypoint_2.npy")
image1_kp = image1_kp.astype(int)
image2_kp = image2_kp.astype(int)


# x_min__left = image1_kp[2][0]
# y_max_left = image1_kp[0][1]
# x_max_right = image2_kp[0][1]
# y_min_right = image2_kp[1][1]


# image2_kp_shift = np.copy(image2_kp)
#
# for i in range(0,len(image2_kp_shift)):
#     image2_kp_shift[i][0] = image2_kp_shift[i][0]+width
def point_inside_polygon(point, poly, include_edges=True):
    '''
    Check if point (x,y) is inside polygon poly.

    poly is N-vertices polygon defined as
    [[x1,y1],...,[xN,yN]] or [[x1,y1],...,[xN,yN],[x1,y1]]
    (function works fine in both cases)

    Geometrical idea: point is inside polygon if horisontal beam
    to the right from point crosses polygon even number of times.
    Works fine for non-convex polygons.
    '''
    n = len(poly)
    inside = False
    x, y = point[0], point[1]
    p1x, p1y = poly[0][0], poly[0][1]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n][0], poly[i % n][1]
        if p1y == p2y:
            if y == p1y:
                if min(p1x, p2x) <= x <= max(p1x, p2x):
                    # point is on horisontal edge
                    # inside = include_edges
                    inside = True
                    break
                elif x < min(p1x,
                             p2x):  # point is to the left from current edge
                    inside = not inside
        else:  # p1y!= p2y
            if min(p1y, p2y) <= y <= max(p1y, p2y):
                xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x

                if x == xinters:  # point is right on the edge
                    # inside = include_edges
                    inside = True
                    break

                if x < xinters:  # point is to the left from current edge
                    inside = not inside

        p1x, p1y = p2x, p2y

    return inside


def reference_point(src_point, matrix_tranform):
    x_src, y_src = src_point[0], src_point[1]
    x_des = (matrix_tranform[0][0] * x_src + matrix_tranform[0][1] * y_src +
             matrix_tranform[0][2]) / (
                    matrix_tranform[2][0] * x_src + matrix_tranform[2][
                1] * y_src + matrix_tranform[2][2])
    y_des = (matrix_tranform[1][0] * x_src + matrix_tranform[1][1] * y_src +
             matrix_tranform[1][2]) / (
                    matrix_tranform[2][0] * x_src + matrix_tranform[2][
                1] * y_src + matrix_tranform[2][2])
    return [x_des, y_des]


def is_near(point1, point2, distance=10):
    return math.sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1],
                                                         2)) > distance


def check_overlap(human_locations, poly_left, poly_right, matrix_tranform):
    ID = 1
    identity = {}
    human_in_left, human_in_right = [], []
    human_overlap_left, human_overlap_right = [], []
    for human_location in human_locations:
        if human_location[0] >= width:
            human_in_right.append(human_location)
        else:
            human_in_left.append(human_location)
    for human_location in human_in_right:
        human_location_shift = human_location
        human_location_shift[0] = human_location[0]-width
        if not point_inside_polygon(human_location_shift, poly_right):
            identity[ID] = [human_location]
            ID = ID + 1
        else:
            human_overlap_right.append(human_location)
    for human_location in human_in_left:
        if not point_inside_polygon(human_location, poly_left):
            identity[ID] = [human_location]
            ID = ID + 1
        else:
            human_overlap_left.append(human_location)
    while len(human_overlap_right) > 0:
        human_location = human_overlap_right.pop()
        same_human = [human_location]
        # shift location
        human_location[0] = human_location[0] - width
        human_reference_location = reference_point(human_location,
                                                   matrix_tranform)
        for human_location_dst in human_overlap_left:
            if is_near(human_location_dst,human_reference_location,15):
                same_human.append(human_location_dst)
                human_overlap_left.remove(human_location_dst)
        identity[ID] = same_human
        ID = ID +1
    while len(human_overlap_left)>0:
        identity[ID] = [human_overlap_left.pop()]
        ID = ID +1
    return identity
def main(argv1, argv2):
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    H = np.load("H_matrix.npy")

    src_point = (300, 410)
    src_point_np = np.concatenate([np.array(src_point), np.array([1])])
    src_point_np = np.array([src_point_np]).T
    color = (0, 0, 255)

    # draw point in right hand side
    cv2.circle(img2, src_point, 5, color, -1)
    img2 = drawROI(img2, image2_kp)
    # cv2.imshow("right", img2)

    dst_point_np = np.matmul(H, src_point_np)
    # dst_point_np = cv2.perspectiveTransform(np.array([np.array([np.array(src_point)])]).astype(
    #     np.float32), H)
    # dst_point_np = cv2.warpPerspective(
    #         np.array([np.array([np.array(src_point)])]).astype(
    #                     np.float32), H, (1000,1000))
    print(dst_point_np)
    dst_point_np[0][0] = dst_point_np[0][0] / dst_point_np[2][0]
    dst_point_np[1][0] = dst_point_np[1][0] / dst_point_np[2][0]
    print(dst_point_np.shape)
    # cv2.imshow('dst', dst_point_np)
    # cv2.waitKey(0)
    dst_point = (int(dst_point_np[0][0]), int(dst_point_np[1][0]))
    # print(dst_point)

    # draw point in left hand side
    cv2.circle(img1, dst_point, 5, color, -1)
    img1 = drawROI(img1, image1_kp)
    # cv2.imshow("left", img1)

    result = np.zeros((height, width * 2, 3)).astype(np.uint8)
    result[:, 0:width, :] = img1
    result[:, width:, :] = img2
    # result = drawROI(result, image2_kp_shift)
    cv2.imshow('concat', result)
    # final=blending(img1, img2, H)
    # cv2.imwrite("panorama.png", final)
    # cv2.imshow('panorama', final)
    cv2.waitKey(0)


if __name__ == '__main__':
    # main("image_left.png", "image_right.png")
    matrix_tranform = np.load('H_matrix.npy')
    # do detect
    human_locations = []
    # do identity
    result = check_overlap(human_locations,image1_kp, image2_kp, matrix_tranform)