import cv2, numpy as np

face_cascade = cv2.CascadeClassifier("images/haar_files/haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("images/haar_files/haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("images/haar_files/haarcascade_mcs_mouth.xml")

def mode_choice():
    print("0. 종료")
    print("1. 저장된 이미지에서 얼굴 검출")
    print("2. 캠에서 얼굴 검출")
    mode = input("원하시는 모드를 선택해주세요 : ")
    return mode

def preprocessing_image(no):
    image = cv2.imread('images/face/%02d.jpg'%no, cv2.IMREAD_COLOR)
    if image is None: return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return image, gray

def faces_detect(image, gray):
    faces = face_cascade.detectMultiScale(gray, 1.2, 4, 0, (10, 10))
    if len(faces):
        for (x, y, w, h) in faces:
            face_image = image[y:y + h, x:x + w]
            eyes_image = image[y:y + h * 2 // 3, x:x + w]
            mouth_image = image[y + h * 7 // 10:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(eyes_image, 1.1, 2, 0, (10, 5))
            mouth = mouth_cascade.detectMultiScale(mouth_image, 1.15, 7, 0, (3, 1))
            if len(eyes) == 2:
                for ex, ey, ew, eh in eyes:
                    center = (x + ex + ew // 2, y + ey + eh // 2)
                    cv2.circle(image, center, 10, (0, 255, 0), 2)
            else:
                print("눈 미검출")

            if len(mouth):
                for mx, my, mw, mh in mouth:
                    cv2.rectangle(image, (x + mx, y + my + h * 7 // 10), (x + mx + mw, y + my + mh + h * 7 // 10),
                                  (0, 0, 255))
            else:
                print("입 미검출")
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    else:
        print("얼굴 미검출")

def image_mode():
    while True :
        print("\n1~59번은 한명 60~63번은 여러명입니다.")
        num = int(input("원하시는 번호를 입력해주세요 : "))
        if 1 <= num <= 63 : break
        else : print("\n올바르지 못한 번호입니다. 정확한 번호를 입력해 주시기 바랍니다.")
    image, gray = preprocessing_image(num)
    if image is None: raise Exception("영상파일 읽기 에러")
    faces_detect(image, gray)
    cv2.imshow("image", image)
    print("\n아무 키를 입력하시면 창이 없어집니다.\n")
    cv2.waitKey()
    cv2.destroyAllWindows()

def preprocessing_cam(frame):
    image = frame
    if image is None: return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return image, gray

def cam_mode():
    print("\n잠시만 기다려 주세요\n")
    capture = cv2.VideoCapture(0)
    if capture.isOpened() == False:
        raise Exception("카메라 연결 안됨")
    title = "cam_face_detect"
    while True :
        ret, frame = capture.read()
        if not ret : break
        image, gray = preprocessing_cam(frame)
        faces_detect(image, gray)
        cv2.imshow(title, frame)
        if cv2.waitKey(30) >= 0 : break
    cv2.destroyAllWindows()
    capture.release()

while True :
    mode = mode_choice()
    if mode == '0':
        print("프로그램을 종료합니다.")
        break
    elif mode == '1': image_mode()
    elif mode == '2': cam_mode()
    else : print("\n번호를 잘못 누르셨습니다. 다시 한번 입력해 주시기 바랍니다.\n")