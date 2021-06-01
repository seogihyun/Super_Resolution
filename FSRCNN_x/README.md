
# FSRCNN
FSRCNN은 SRCNN의 단점을 보완하고 가속시킨 네트워크로, 기존 SRCNN과 같은 정확도를 유지하면서 속도는 최대 40배 가속 됨
 
## Abstract

![fsrcnn](https://user-images.githubusercontent.com/72849922/120250903-bcab9880-c2ba-11eb-858a-adcb153154d4.png)  



FSRCNN은 상대적으로 얕은 네트워크를 가지고 있어 각 구성 요소의 효과를 더 쉽게 배울 수 있음

또한 FSRCNN-s 모델은 FSRCNN을 경량화시킨 모델로, 미미한 차이의 성능 저하를 감수하고 속도를 더욱 향상시켰으며 이는 SRCNN 보다 훨씬 빠른 속도를 보임




![0_jPCy664hkmSJiVkP](https://user-images.githubusercontent.com/72849922/120270299-55a2d980-c2e4-11eb-9dc6-0c3a1298cc6d.png)



## FSRCNN structure

- **1단계 : 특징 추출(feature extraction)**
  - num_channels = 1
  - 원본 이미지를 그대로 입력 데이터로 사용
  - 5x5 필터
  - d = 56
  - s = 12
  - m = 4
- **2단계 : 축소(shrinking)**
  - 1x1 필터
- **3단계 : 매핑(mapping)**
  - SR성능에 영향을 미치는 가장 중요한 부분(매핑층의 개수 & 필터의 크기)
  - 여러 층의 3x3 필터
- **4단계 : 확장(expanding)**
  - 축소 이전의 채널의 크기와 동일하게 확장
  - 1x1 필터
- **5단계 : 업스케일링(deconvolution)**
  - Convolutional Transpose 사용하여 업스케일링
  - 업 스케일링 요소에 관계 없이 모든 이미지가 기존 레이어의 가중치와 편향 값 공유
  - 학습시간 감소
  - 9x9 필터


## From SRCNN to FSRCNN
![다운로드](https://user-images.githubusercontent.com/72849922/120271517-72d8a780-c2e6-11eb-85ca-23390e02c04d.png)


## Experiments
![다운로드 (1)](https://user-images.githubusercontent.com/72849922/120271693-bcc18d80-c2e6-11eb-8b8a-208ba6c30feb.png)


| 유형 | 링크 |
|---|:---:|
| 원본 논문 | [Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/abs/1608.00367) |
| 논문 정리 |[FSRCNN](https://github.com/KHS0616/SuperResolution/blob/master/Paper/FSRCNN.md), [FSRCNN Review](https://sofar-sogood.tistory.com/entry/FSRCNN-%EB%A6%AC%EB%B7%B0-Accelerating-the-Super-Resolution-Convolutional-Neural-Network-ECCV-16)|


----------------------------------------------------------------------------------------------

# FSRCNN-x
필자는 FSRCNN 모델을 경량화 시킨 FSRCNN-s 모델과 Optimal-FSRCNN 모델을 참고하여 FPGA에서 작동 가능한 초경량화된 FSRCNN-x 모델을 만들었음

## FSRCNN-x structure
- **1단계 : 특징 추출(feature extraction)**
  - num_channels = 1
  - 원본 이미지를 그대로 입력 데이터로 사용
  - 3x3 필터
  - d = 23
  - s = 12
  - m = 2
- **2단계 : 축소(shrinking)**
  - 1x1 필터
- **3단계 : 매핑(mapping)**
  - SR성능에 영향을 미치는 가장 중요한 부분(매핑층의 개수 & 필터의 크기)
  - 여러 층의 3x3 필터
- **4단계 : 확장(expanding)**
  - 축소 이전의 채널의 크기와 동일하게 확장
  - 1x1 필터
- **5단계 : 업스케일링(deconvolution)**
  - Convolutional Transpose 사용하여 업스케일링
  - 업 스케일링 요소에 관계 없이 모든 이미지가 기존 레이어의 가중치와 편향 값 공유
  - 학습시간 감소
  - 3x3 필터



| 유형 | 링크 |
|---|:---:|
| 참고 논문 | [Deep Learning-based Real-Time Super-Resolution Architecture Design](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002699585) |
| 참고 논문 |[Accelerating the Super-Resolution Convolutional Neural Network](https://github.com/KHS0616/SuperResolution/blob/master/Paper/FSRCNN.md)| 

