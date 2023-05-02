import cv2
import numpy as np

prototxt = "SSD_data/deploy.prototxt"
caffemodel = "SSD_data/res10_300x300_ssd_iter_140000_fp16.caffemodel"
detector = cv2.dnn.readNet(prototxt, caffemodel)

capture = cv2.VideoCapture(0)
while True :
    ret, frame = capture.read()
    if not ret : break
    image = frame
    (h, w) = image.shape[:2]
    target_size = (300, 300)
    input_image = cv2.resize(image, target_size)
    imageBlob = cv2.dnn.blobFromImage(input_image)
    detector.setInput(imageBlob)
    detections = detector.forward()

    results = detections[0][0]
    threshold = 0.8
    for i in range(0, results.shape[0]):
        conf = results[i, 2]
        if conf < threshold:
            continue

        box = results[i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')

        cv2.putText(image, str(conf), (startX, startY-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.imshow('image', image)
    if cv2.waitKey(30) >= 0 : break
cv2.destroyAllWindows()
capture.release()