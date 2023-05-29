import numpy as np
import cv2
import matplotlib.pyplot as plt

def init(): # 초기 설정
    width = 120; height = 150; rate = 0.95; input_img_num = 310; test_img_num = 93
    return width, height, rate, input_img_num, test_img_num
    # 사진 너비, 사진 높이, 고유값 뽑을 비율, input 이미지 개수, output 이미지 개수

def diffent_image(): # 차영상 구하기
    average_img = np.zeros((height, width), dtype='float32') # float32 쓰는 이유는 나누기가 있기 때문에
    for i in range(input_img_num):
        img = cv2.imread('face_img/train/train%03d.jpg' % i, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (width, height))
        average_img += np.array(img, dtype='float32')
    average_img /= input_img_num
    # average_img = average_img.astype('uint8') # 평균영상 출력하기
    # cv2.imshow('average_img', average_img)
    # cv2.moveWindow('average_img', 300, 300)
    # average_img = average_img.astype('float32')
    diff_img = np.zeros((1, height * width), dtype='float32')
    for i in range(input_img_num):
        img = cv2.imread('face_img/train/train%03d.jpg' % i, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (width, height)) - average_img
        img = np.resize(img, (1, width * height)) # 영상을 1,18000로 변환
        diff_img = np.append(diff_img, img, axis=0)
    diff_img = np.delete(diff_img, 0, axis=0) # 맨 처음은 0으로 채워져 있으니 제거
    return diff_img, average_img

def eigenvalue_eigenvector():
    covariance_matrix = np.cov(diff_img) # 공분산 행렬 구하기
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)  # 고유값, 고유벡터 추출
    idx = eigenvalues.argsort()[::-1]  # 고유값 내림차순으로 순서 추출
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # plt.plot(eigenvalues) # 고유값 그래프로 보기
    # plt.show()
    eigenvalues_sum = 0
    for i, v in enumerate(eigenvalues):
        eigenvalues_sum += v
        if eigenvalues_sum / eigenvalues.sum() >= rate: # 처음에 정한 비율만큼만 사용하기
            eigenvalues_count = i + 1                   # 숫자가 큰 고유값이 의미가 큼
            break
    return eigenvalues_count, eigenvectors

def transform_matrix(): # 변환 행렬 및 특징 행렬
    transform_matrix = np.zeros((width * height, 1), dtype='float32')
    for i in range(eigenvalues_count):
        tem = (diff_img.T @ eigenvectors[:, i:i+1]).reshape(width*height, 1)
        transform_matrix = np.append(transform_matrix, tem / np.linalg.norm(tem), axis=1) # 정규화
    transform_matrix = np.delete(transform_matrix, 0, axis=1)  # 18000,85
    feature_value = transform_matrix.T @ diff_img.T  # 85, 310
    return feature_value, transform_matrix

###################################################
def test(): # 자기가 원하는 이미지 넣어서 테스트 하는 단계
   while True :
       print("0번을 누르면 종료됩니다.")
       print("1번을 누르면 원하시는 번호의 사진과 비교할 수 있습니다.\n")
       n = int(input())
       if n == 0: break
       elif n == 1:
           i = int(input("비교를 원하시는 사진의 번호를 입력해 주세요(0~92)\n"))
           if i < 0 or i > 92:
               print("올바르지 못한 번호입니다. 다시 입력해 주시기 바랍니다.\n")
               continue
       else :
           print("올바르지 않은 번호입니다. 다시 입력해 주시기 바랍니다.\n\n")
           continue
       input_img = cv2.imread('face_img/test/test%03d.jpg' % i, cv2.IMREAD_COLOR)
       input_img = cv2.resize(input_img, (300, 300))
       cv2.imshow(f"input {i}", input_img) # 입력영상 출력
       cv2.moveWindow(f'input {i}', 400, 300)

       input_img = cv2.imread('face_img/test/test%03d.jpg' % i, cv2.IMREAD_GRAYSCALE)
       input_img = cv2.resize(input_img, (width, height))
       test_diff_img = np.array(input_img) - average_img
       test_diff_img = np.reshape(test_diff_img, (height * width, 1))
       test_feture = transform_matrix.T @ test_diff_img  # 85,18000 @ 18000,1 # 입력영상 특징행렬 만들기

       min_value = None;
       num = 0
       for j in range(input_img_num):   # 유클리드 거리법을 이용하여 제일 가깝다고 생각되는 이미지 찾기
           value_sum = 0
           for k in range(eigenvalues_count):
               value_sum += (feature_value[k, j] - test_feture[k, 0]) ** 2
           if j == 0 or min_value > value_sum:
               min_value = value_sum
               num = j
       img = cv2.imread('face_img/train/train%03d.jpg' % num)
       img = cv2.resize(img, (300, 300))
       cv2.imshow(f"result {num}", img)           # 출력영상
       cv2.moveWindow(f'result {num}', 800, 300)
       cv2.waitKey()
       cv2.destroyAllWindows()

width, height, rate, input_img_num, test_img_num = init()
diff_img, average_img = diffent_image()
eigenvalues_count, eigenvectors = eigenvalue_eigenvector()
feature_value, transform_matrix = transform_matrix()
test()