import numpy as np
import cv2

width = 120     # 사진 너비
height = 150    # 사진 높이
rate = 0.95     # 사용할 고윳값 비율
eigenvalues_sum = 0
eigenvalues_count = 0
learning_img_num = 310  # 학습 이미지 개수
output_img = 93
average_img = np.zeros((height, width), dtype='float32')
diff_img = np.zeros((1, height*width), dtype='float32')

for i in range(learning_img_num):
    learning_img = cv2.imread('train/train%03d.jpg'%i, cv2.IMREAD_GRAYSCALE)
    learning_img = cv2.resize(learning_img, (width, height))
    average_img += np.array(learning_img, dtype='float32')
average_img /= learning_img_num

# average_img = average_img.astype('uint8') 평균영상 출력하기
# cv2.imshow('average_img', average_img)
# cv2.waitKey()

for i in range(learning_img_num):
    learning_img = cv2.imread('train/train%03d.jpg' % i, cv2.IMREAD_GRAYSCALE)
    learning_img = cv2.resize(learning_img, (width, height)) - average_img
    learning_img = np.resize(learning_img, (1, width*height))
    diff_img = np.append(diff_img, learning_img, axis=0)

diff_img = np.delete(diff_img, 0, axis=0) # 맨 처음은 0으로 채워져 있으니 제거
covariance_matrix = np.cov(diff_img) # 공분산 행렬
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix) # 고유값, 고유벡터 추출
idx = eigenvalues.argsort()[::-1] # 고유값 내림차순으로 순서 추출
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

for i, v in enumerate(eigenvalues):
    eigenvalues_sum += v
    if eigenvalues_sum / eigenvalues.sum() >= rate:
        eigenvalues_count = i + 1
        break

transform_matrix = np.zeros((width*height, 1), dtype='float32')

for i in range(eigenvalues_count):
    tem = (diff_img.T @ eigenvectors[:, i:i+1]).reshape(width*height, 1)
    transform_matrix = np.append(transform_matrix, tem / np.linalg.norm(tem), axis=1)
transform_matrix = np.delete(transform_matrix, 0, axis=1) # 18000,85
feature_value = transform_matrix.T @ diff_img.T # 85, 310

print(feature_value.shape)

for i in range(output_img):
    input_img = cv2.imread('test/test%03d.jpg'%i, cv2.IMREAD_COLOR)
    input_img = cv2.resize(input_img, (width, height))
    input_img = cv2.imshow("input_img", input_img)
    cv2.moveWindow('input_img', 500, 100)
    input_img = cv2.imread('test/test%03d.jpg'%i, cv2.IMREAD_GRAYSCALE)
    input_img = cv2.resize(input_img, (width, height))

    test_diff_img = np.array(input_img) - average_img
    test_diff_img = np.reshape(test_diff_img, (height * width, 1))

    test_feture = transform_matrix.T @ test_diff_img # 85,18000 @ 18000,1

    min_value = None; num = 0
    for j in range(learning_img_num):
        value_sum = 0
        for k in range(eigenvalues_count):
            value_sum += (feature_value[k, j] - test_feture[k, 0]) ** 2
        if j == 0 or min_value > value_sum:
            min_value = value_sum
            num = j
    img = cv2.imread('train/train%03d.jpg'%num)
    img = cv2.resize(img, (width, height))
    cv2.imshow("img", img)
    cv2.moveWindow('img', 100, 100)
    cv2.waitKey(1000)