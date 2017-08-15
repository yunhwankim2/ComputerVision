# 동영상 파일을 입력받아, shot boundary detection 과 key frame selection 을 실행해서,
# 각 key frame 을 사진 파일로 저장하는 것이 목표.

import cv2
import os
import numpy as np
import pandas as pd


# read the video file

filename = ""

videoCapture = cv2.VideoCapture(filename)

fps = videoCapture.get(cv2.CAP_PROP_FPS)    # frame per second
length = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))    # number of frames
size = (
    int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
    int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

success, frame = videoCapture.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# calculate histogram differences

HD = np.zeros(length-1)   # dissimilarity measure is histogram difference
# 길이가 하나 작으니, 원래 프레임 번호와 HD 프레임 번호를 잘 구분할 것.

# 0번 프레임과 1번 프레임 사이의 histogram difference 를 계산하기 위해 
# 0번 프레임의 히스토그램을 먼저 생성
f_0 = cv2.calcHist([frame], [0, 1, 2], None, [64, 64, 64], [0,256, 0,256, 0,256])
# cv2.calcHist([image], channel, mask, bins, range)

f_0 = cv2.normalize(f_0, cv2.NORM_MINMAX).flatten()

# histogram difference 들을 계산해서 HD 에 순서대로 저장
for i in range(length):
    success, frame = videoCapture.read()
    if success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        break
    
    f_1 = cv2.calcHist([frame], [0, 1, 2], None, [64, 64, 64], [0,256, 0,256, 0,256])
    f_1 = cv2.normalize(f_1, cv2.NORM_MINMAX).flatten()
    
    # calculate the  histogram difference
    difference = cv2.compareHist(f_0, f_1, cv2.HISTCMP_CHISQR)
            # HistCompMethods :
            # cv2.HISTCMP_CORREL = 0, 
            # cv2.HISTCMP_CHISQR = 1, 
            # cv2.HISTCMP_INTERSECT = 2, 
            # cv2.HISTCMP_BHATTACHARYYA = 3, 
            # cv2.HISTCMP_HELLINGER = HISTCMP_BHATTACHARYYA, 
            # cv2.HISTCMP_CHISQR_ALT = 4, 
            # cv2.HISTCMP_KL_DIV = 5 
            # LARGER value indicates higher similarity (Correlation and Intersection). 
            # SMALLER value indicates higher similarity (Chi-Squared and Hellinger).
    HD[i] = difference
    
    # change histograms
    f_0 = f_1


#############################

# 이제 histogram difference 가 확보됨.

shot_beginning_frame = []   # 샷의 시작점이 되는 프레임 번호를 저장할 리스트

# 기본적으로 아래의 문헌에 나온 방법을 사용.
# Yusoff, Y., Christmas, W. J., & Kittler, J. (2000). Video Shot Cut Detection using Adaptive Thresholding. Presented at the British Machine Vision Conference, British Machine Vision Association. http://doi.org/10.5244/C.14.37
# 4. The adaptive thresholding scheme 참조.
# The middle sample represents a shot change if the conditions below are simultaneously satisfied:
# 1. The middle sample is the maximum in the window
# 2. The middle sample is greater than max(mu_left + Td*SD_left, mu_right + Td*SD_right)
# where Td is given a value of 5.

half_w_size = 10    # half window size
Td = 5              # constant


for i in range(half_w_size, length-half_w_size+1):
    begin = i - half_w_size
    end = i + half_w_size + 1
    window = HD[begin:end]
    local_max = np.max(window)
    if HD[i] < local_max:
        continue
    else:        
        left_window = HD[begin:i]
        right_window = HD[i+1:end]
        mu_left = np.mean(left_window)
        mu_right = np.mean(right_window)
        SD_left = np.std(left_window)
        SD_right = np.std(right_window)
        th_left = mu_left + (Td * SD_left)
        th_right = mu_right + (Td * SD_right)
        if HD[i] > max([th_left, th_right]):
            shot_beginning_frame.append(i)


#############################

# 이제 shot beginning frame 이 확보됨. 이제는 key frame 을 설정할 차례.        
# shot beginning frame 과 그 다음 시작 frame 사이의 중앙 frame 을 key frame 으로 선택함.

key_frame = []  # key frame 에 해당하는 프레임 번호를 저장할 리스트

a = 0
b = 1
for i in range(length):
    if i in shot_beginning_frame:
        b = i - 1
        key_frame_number = int((a + b) / 2)
        key_frame.append(key_frame_number)
        a = i

###################################

# 키 프레임에 해당되는 사진을 저장

kf_number = 1
videoCapture = cv2.VideoCapture(filename)

for i in range(length):
    success, frame = videoCapture.read()
    if i in key_frame:
        if not success:
            break
        
        dir_name = filename.split('.')[0]
                        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        image_name = "./" + dir_name + "/" + dir_name + "_" + str(kf_number) + ".jpg"
        cv2.imwrite(image_name, frame)
        kf_number += 1
    








