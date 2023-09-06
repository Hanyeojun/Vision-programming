import numpy as np, cv2, ntpath
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
from PIL import Image, ImageFont, ImageDraw

# 학습 단계
arr = np.array([], dtype=np.float32)    # 학습 영상 저장 배열
avg = np.zeros((120*150), dtype=np.float32) # 평균 영상 저장 배열
sub = np.array([], dtype=np.float32) # 차 영상 저장 배열

for no in range(0, 310): # 학습 영상의 개수만큼 for문 반복
    img = cv2.imread("face_img/train/train%003d.jpg" %no, cv2.IMREAD_GRAYSCALE) # 학습 영상 읽어오기
    reimg = cv2.resize(img, dsize=(120, 150)) # 학습 영상 크기 150*120으로 변경
    dim1 = reimg.reshape(-1)    # 1차원 변환
    arr = np.append(arr, dim1) # arr에 1차원 행렬로 변환하여 저장
    avg += dim1 # 학습 영상 행렬 누적 합 저장

avg = avg / 310 # 학습 영상 평균 계산후 저장
avg = avg.reshape(150, 120) # 학습 영상을 2차원 배열로 변환
cv2.imwrite("avg.jpg", avg) # 평균 영상 파일 저장
avg = avg.reshape(-1) # 평균 행렬 1차원 변환

num = 0 # 한 개의 학습 영상을 가져오기 위한 변수
for i in range(310):
    tmp = arr[num:num+18000]-avg[:18000] # 학습 영상과 평균 영상의 차 계산
    sub = np.append(sub, tmp) # 계산한 차 값을 sub 배열에 저장
    num += 18000 # 다음 학습 영상으로 이동

sub = sub.reshape(310, 18000) # 2차원. (310, 18000)
subT = sub # sub의 transpose 행렬
sub = subT.transpose()   #차 영상 행렬 전치

cov = np.dot(subT, sub) # 공분산 행렬 계산
eigenval, snapEigenvec = np.linalg.eig(cov) # 고유값, 스냅샷 고유 벡터 계산
orgEigenvec = np.array([], np.float32)
# 차 행렬:(18000, 310), 스냅 고유 벡터:(310, 1)

for i in range(310): # 원래의 고유 벡터 계산
    #snapEigenvec = snapEigenvec.transpose() # (310, 310), 원래 세로 벡터를 가로 방향 전환
    tmp = snapEigenvec[i] # 스냅 고유 벡터 한 행 가져옴
    tmp = tmp.reshape(-1,1) # 세로 행렬로 변환, (310, 1)
    t = sub@tmp # t : (18000, 1)
    orgEigenvec = np.append(orgEigenvec, t) # (5580000,)
orgEigenvec = orgEigenvec.reshape(310, 18000) # 행 하나가 한 고유 벡터

# 고유 벡터 정규화
for i in range(310):
    eigenvecNorm = np.linalg.norm(orgEigenvec[i])  # 고유 벡터의 노름 계산
    orgEigenvec[i] = orgEigenvec[i] / eigenvecNorm  # 고유 벡터 정규화
#orgEigenvec = orgEigenvec.reshape(18000, 310)
#print(orgEigenvec.shape)
orgEigenvec = orgEigenvec.T
#print(orgEigenvec.shape)

# 고유값 분포 그래프 그리기
# pyplot.plot(eigenval) # 고유값에 대한 그래프 그리기
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_powerlimits((-1, 1))
# pyplot.gca().yaxis.set_major_formatter(formatter) # 현재 축에 대한 정보를 가져옴
# pyplot.show() # 그래프 표시

k = 0.98
sortEigenval = np.sort(eigenval)[::-1] # 고유값 내림차순 정렬
count = eigenval.sum()*k # 고유값 전체 합 * 0.95
sortEigsum = 0 # 고유값 누적합 저장
v = 0 # 최소 고유값 개수 v

# 최소 고유값 개수 계산
for i in range(0, 310): # 310개의 고유값 개수만큼 반복
    if count <= sortEigsum: # 누접합이 고유값 전체 합*0.95보다 커지면 for문 탈출
        break
    else:
        sortEigsum += sortEigenval[i] # 누접합 저장
        v += 1 # 최소 고유값의 개수 1 증가

orgEigenvecT = orgEigenvec.transpose() # 고유벡터 transpose(310, 18000)
idx = np.array([], np.float32) # 고유벡터의 index 가져오기 위한 배열
for i in range(v): # 최소 고유값 개수만큼 반복
    for j in range(310): # 310개의 교유값 중에서
        if sortEigenval[i] == eigenval[j]: #
            idx = np.append(idx, j) # 큰 고유값의 인덱스부터 저장됨

transferM = np.array([], np.float32) # 변환행렬

for i in range(v):
    transferM = np.append(transferM, orgEigenvecT[int(idx[i])]) # (1, 18000) 고유벡터 저장
transferM = transferM.reshape(v, 18000) # v개의 고유벡터 저장

featureM = np.array([], np.float32) # 특징 행렬
# 차 행렬 : (18000, 310), 변환행렬T = (v, 18000)

for i in range(310):
    tmp = subT[i] # 차 영상 하나 tmp에 저장
    tmp = tmp.reshape(-1, 1) # (18000, 1)로 형태 변환
    #tmp = tmp.flatten()
    xi = transferM@tmp # 각 영상의 특징 벡터 계산
    print(xi.shape)
    featureM = np.append(featureM, xi) # 계산한 특징 벡터 저장
print(featureM.shape)
featureM = featureM.reshape(310, v)
# 특징 벡터 하나가 (v, 1)이며 이 특징 벡터를 featureM 행렬에 append하면 가로 방향으로 들어가므로
# reshape하여 행 하나가 특징 벡터 하나를 나타내도록 저장함

# Test 단계
for kkk in range(8):
    n = int(input("테스트 영상 입력(0 ~ 92) : "))
    subsrc = np.array([], np.float32)  # 차 영상
    res = 0 # 결과 영상 index 저장
    src = cv2.imread("face_img/test/test%003d.jpg" %n, cv2.IMREAD_GRAYSCALE)  # test 영상
    resrc = cv2.resize(src, dsize=(120, 150))  # test 영상 크기 조절
    resrc = resrc.reshape(-1)  # 입력 영상 1차원 변환

    subsrc = resrc - avg  # 차영상 : 입력 영상 - 평균 영상

    featsrc = transferM @ subsrc  # 변환 행렬과 차 영상 곱

    min_dist = 100000000000 # 최소값 계산 위한 배열
    dst = np.array([], np.float32) # 유클리드 거리 계산값 배열

    for i in range(310):
        dst = np.linalg.norm(featureM[i] - featsrc) # 유클리드 거리 계산.
        if dst < min_dist: # 최소 거리 계산. 현재 최소 거리가 이전 최소 거리보다 작다면
            min_dist = dst # 현재 최소 거리를 dst 배열에 저장
            res = i # 현재 최소 거리 영상의 index 저장
    file_nlist = 'Ans_'
    file_nlist += ntpath.basename('face_img/train/train%003d' %res)

    res_img = cv2.imread("face_img/train/train%003d.jpg" %res, cv2.IMREAD_GRAYSCALE)
    src = cv2.resize(src, dsize=(240, 300))
    res_img = cv2.resize(res_img, dsize=(240, 300))
    cv2.imshow("Test_img", src) # 테스트 영상 출력
    cv2.imshow(file_nlist, res_img) # 결과 영상 출력
    cv2.waitKey(0)
    cv2.destroyAllWindows()
