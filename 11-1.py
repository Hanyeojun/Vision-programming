import cv2, numpy as np
def preprocessing(no): # 파일 이름의 번호를 인자로 받음
    image = cv2.imread('images/face/%02d.jpg' %no, cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray) # 히스토그램 평활화
    return image, gray # 컬러 영상, 명암도 영상 반환

# detectMultiScale(객체 검출 대상 행렬, 반환되는 검출 객체 사각형, 영상 크기 감소에 대한 규정,
# 이웃 후보 사각형 개수, 과거 함수에서 사용하던 flag, 가능한 객체 최소 크기, 가능한 객체 최대 크기)

face_cascade = cv2.CascadeClassifier("images/Haar_files/haarcascade_frontalface_alt2.xml")  # 얼굴 찾음
eye_cascade = cv2.CascadeClassifier("images/Haar_files/haarcascade_eye.xml")  # 눈 찾음
mouth_cascade = cv2.CascadeClassifier("images/Haar_files/haarcascade_mcs_mouth.xml")  # 입 찾음

ititle, ctitle = "image", "View From Camera"
flag = int(input("실행할 기능 선택. 이미지 파일 : 숫자 1 입력, 카메라 영상 : 숫자 2 입력\n"))

global yesno
yesno = False

def facescascading(image):
    global yesno
    faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))
    if len(faces):
        yesno = True
        for i in range(len(faces)):
            x, y, w, h = faces[i]
            face_image1 = image[y:y + h * 2 // 3, x:x + w]
            face_image2 = image[y + h // 2:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face_image1, 1.05, 4, 0, (10, 15))
            mouths = mouth_cascade.detectMultiScale(face_image2, 1.08, 20, 0, (10, 20))
            if len(eyes) == 2:
                for ex, ey, ew, eh in eyes:
                    center = (x + ex + ew // 2, y + ey + eh // 2)
                    cv2.circle(image, center, 10, (0, 255, 0), 2)
            else:
                print("눈 미검출")

            if len(mouths) == 1:
                for mx, my, mw, mh in mouths:
                    cv2.rectangle(image, (x + mx, y + my + h // 2), (x + mx + mw, y + my + mh + h // 2), (0, 0, 255), 2)
            else:
                print("입 미검출")

            cv2.rectangle(image, faces[i], (255, 0, 0), 1)
    else:
        yesno = False
        print("얼굴 미검출")
    return image

if flag == 1:
    num = int(input("출력할 사진 번호 입력.(한 명 : 1~59, (여러명 : 60~62) => "))
    img, gray = preprocessing(num)

    if img is None:
        raise Exception("이미지 파일 읽기 에러")
    facescascading(img)
    if yesno:
        cv2.imshow(ititle, img)
    else:
        cv2.imshow(ititle, img)
    cv2.waitKey(0)

elif flag == 2:
    capture = cv2.VideoCapture(0)  # 0번 카메라 연결
    if capture.isOpened() == False:  # 카메라 연결 안될 때
        raise Exception("카메라 연결 안됨")

    while True:
        # 카메라 영상 받기, 잘 작동하면 ret = true
        ret, frame = capture.read()
        if not ret:  # frame 못 받으면 종료
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        if frame is None:
            raise Exception("영상 파일 읽기 에러")

        frame = facescascading(frame)

        # 윈도우에 카메라 입력 영상 출력
        cv2.imshow(ctitle, frame)

        if cv2.waitKey(30) >= 0:  # 스페이스바 누르면 종료
            break
    capture.release()  # 카메라 연결 해제

cv2.destroyAllWindows()