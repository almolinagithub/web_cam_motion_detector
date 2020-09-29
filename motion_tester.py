
import cv2, time
from datetime import  datetime
import pandas as pd

df = pd.DataFrame(columns=["start","end"])
first_frame=None
times = [None,None]
status_list = [0]
video = cv2.VideoCapture(0)
time.sleep(0.5)

while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21),0)

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)

    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2 )
    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 15000:
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x, y), (x + w, y + h), (0,255,0), 3)
            #	img	=	cv.rectangle(	img, pt1, pt2, color[, thickness[, lineType[, shift]]]	)
    status_list.append(status)

    if status_list[-1]== 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1]== 0 and status_list[-2] == 1:
        times.append(datetime.now())

    status_list =   status_list[-2:]
    #df.to_csv(times)

    cv2.imshow('gray frame',gray)
    cv2.imshow('delta frame', delta_frame)
    cv2.imshow('threshold frame', thresh_frame )
    cv2.imshow('color frame', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

print(times)
print(status_list)
for i in range(0,len(times),2):
    df = df.append({
    "start":times[i],
    "end":times[i+1],
    },ignore_index=True)

df.to_csv("times.csv")
video.release()
cv2.destroyAllWindows
