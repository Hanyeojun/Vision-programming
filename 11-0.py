import cv2, numpy as np
def preprocessing(no): # 파일 이름의 번호를 인자로 받음
    image = cv2.imread('images/face/61.jpg', cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray) # 히스토그램 평활화
    return image, gray # 컬러 영상, 명암도 영상 반환

face_cascade = cv2.CascadeClassifier("images/Haar_files/haarcascade_frontalface_alt2.xml") # 얼굴 찾음
eye_cascade = cv2.CascadeClassifier("images/Haar_files/haarcascade_eye.xml") # 눈 찾음
image, gray = preprocessing(61)
if image is None:
    raise Exception("영상 파일 읽기 에러")

faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))
if len(faces):
    for i in range(len(faces)):
        x, y, w, h = faces[i]
        face_image = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_image, 1.15, 5, 0, (25, 20))

        if len(eyes) == 2:
            for ex, ey, ew, eh in eyes:
                center = (x+ex+ew//2, y+ey+eh//2)
                cv2.circle(image, center, 10, (0, 255, 0), 2)
        else:
            print("눈 미검출")

        cv2.rectangle(image, faces[i], (255, 0, 0), 2)
        cv2.imshow("image", image)
else:
    cv2.imshow("image", image)
    print("얼굴 미검출")
cv2.waitKey(0)
cv2.destroyAllWindows()