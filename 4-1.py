import numpy as np
import cv2

green = (0, 255, 0)
oldx = oldy = -1

def onMouse(event, x, y, flags, param):
    global oldx, oldy
    if event == cv2.EVENT_LBUTTONDOWN:
        oldx, oldy = x, y
        print("click", x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        print("up", x, y)
        temp = img[oldy:y, oldx:x]
        rimg = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        cv2.imshow(title2, rimg)
        cv2.rectangle(img, (oldx, oldy), (x, y), green, 1)
        cv2.imshow(title1, img)
        oldx, oldy = x, y

title1= "mouse event1"
title2 = "result"
img = cv2.imread("images/aa.jpg")

cv2.imshow(title1, img)
cv2.setMouseCallback(title1, onMouse)

cv2.waitKey(0)
cv2.destroyAllWindows()

# import numpy as np
# import cv2
# start_x, start_y, drag = -1, -1, 0
# def onMouse(event, x, y, flags, param):
#     global start_x, start_y, drag
#
#     img_result = param
#     green = (0, 255, 0)
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drag = 1
#         start_x = x
#         start_y = y
#     elif event == cv2.EVENT_LBUTTONUP and drag == 1:
#         cv2.rectangle(img_result, (start_x, start_y), (x, y), green, 3)
#         cv2.imshow('img_color', img_result)
#         img_part = img_result[start_y:y, start_x:x, :]
#         img_part1 = cv2.cvtColor(img_part, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('img_part', img_part1)
#         drag = 0
# img_color = cv2.imread("images/aa.jpg")
# cv2.imshow('img_color', img_color)
# cv2.setMouseCallback('img_color', onMouse, img_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()