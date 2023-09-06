import cv2
capture = cv2.VideoCapture(0) # 0번 카메라 연결
if capture.isOpened() == False: # 카메라 연결 안될 때
    raise Exception("카메라 연결 안됨")

# 출력 창 title
title = "View Frame from Camera"

while True:
    # 카메라 영상 받기, 잘 작동하면 ret = true
    ret, frame = capture.read()
    if not ret: # frame 못 받으면 종료
        break
    # 관심 영역의 테두리를 두께 3의 빨간색으로 표시
    cv2.rectangle(frame, (200, 100, 100, 200), (0, 0, 255), 3)
    # 관심 영역 녹색 성분을 100만큼 증가
    frame[100:300, 200:300, 1] += 50

    # 윈도우에 카메라 입력 영상 출력
    cv2.imshow(title, frame)

    if cv2.waitKey(30) >= 0: # 스페이스바 누르면 종료
        break
capture.release() # 카메라 연결 해제