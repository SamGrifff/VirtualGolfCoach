import cv2

path = "my_swing.mp4"
cap = cv2.VideoCapture(path)

print("Opened:", cap.isOpened())
print("Frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame = cap.read()
print("First frame read:", ret)
print("Frame is None:", frame is None)

cap.release()