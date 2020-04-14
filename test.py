import cv2

cap = cv2.VideoCapture("rtsp://khanh29bk:Admin123@192.168.0.11/Src/MediaInput/h264/stream_1/ch_")
# cap = cv2.VideoCapture("http://192.168.0.11:554/videostream.asf?user=khanh29bk&pwd=Admin123")
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, img = cap.read()
    # img = cv2.resize(img, (200, 100))

    # ret = cap.grab()
    # ret, img = cap.retrieve(ret)

    if ret == True:
        # img = cv2.resize(img, (800, 600))
        cv2.imshow('frame1', img)
        # cv2.imwrite('image_left.png', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
