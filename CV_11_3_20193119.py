import cv2
import numpy as np

prototxt = "SSD_data/deploy.prototxt"
caffemodel = "SSD_data/res10_300x300_ssd_iter_140000_fp16.caffemodel"
detector = cv2.dnn.readNet(prototxt, caffemodel) # SSD 신경망 네트워크 생성

capture = cv2.VideoCapture(0)
if capture.isOpened() == False:
    raise Exception("카메라 연결 안됨")

while True:
    ret, frame = capture.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]  # 입력 영상 읽고
    target_size = (300, 300) # (300, 300) 크기로 크기 조정 <- SSD에서 학습시 사용한 영상 크기
    input_image = cv2.resize(frame, target_size)

    # 네트워크 입력 생성, 지정
    imageBlob = cv2.dnn.blobFromImage(input_image) # (300, 300) 크기 영상으로 blob 객체(네트워크 입력 객체) 생성
    detector.setInput(imageBlob)  # blob 객체를 네트워크 입력으로 지정하고 물체 검출
    detections = detector.forward()  # 얼굴 검출 수행

    results = detections[0][0]  # 얼굴 검출기 반환값(1, 1, N, 7) 배열에서 (N, 7) 내용만 복사
    threshold = 0.8  # 신뢰도 0.8 이상만 인정

    startX, startY, endX, endY = 0, 0, 0, 0

    for i in range(0, results.shape[0]):  # results.shape = (N, 7)
        conf = results[i, 2]  # 검출 얼굴 신뢰도(2열에 존재)
        if conf < threshold:  # ignore detections with low confidence
            continue
        # get corner points of face rectangle
        box = results[i, 3:7]*np.array([w, h, w, h])  # 입력 영상에서 검출 얼굴 좌표(좌상단, 우하단) -> 실수
        (startX, startY, endX, endY) = box.astype('int')  # 정수로 변환
        # 관심 영역 추출
        roi = frame[startY:endY, startX:endX]

        # cv2.imwrite('파일 이름', 저장할 이미지)
        cv2.imwrite('CV_11_3_20193119/2_20193119_2.jpg', roi)
        cv2.imshow('roi', roi)
        # 확률 출력하기
        # cv2.putText(image, 출력 문자열, 출력 좌표, 폰트, 폰트 크기, 폰트 색상, 두께)
        cv2.putText(frame, str(conf), (startX, startY-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)  # 얼굴 위치 box 그리기

    cv2.imshow('frame', frame)
    if cv2.waitKey(30) >= 0:
        capture.release()  # 카메라 연결 해제
        cv2.destroyAllWindows()
        break
cv2.waitKey(0)
