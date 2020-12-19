import cv2

url='https://192.168.43.212:8080'+'/video' # url that IP webcam shows
vs = cv2.VideoCapture(url)

while True:
    ret, frame= vs.read()
    if not ret:
        continue
    cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWndows()