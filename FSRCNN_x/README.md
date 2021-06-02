
https://velog.io/@danielseo/Computer-Vision-Super-Light-FSRCNN
# FSRCNN
FSRCNN은 SRCNN의 단점을 보완하고 가속시킨 네트워크로, 기존 SRCNN과 같은 정확도를 유지하면서 속도는 최대 40배 가속 됨
 
## Abstract

![fsrcnn](https://user-images.githubusercontent.com/72849922/120250903-bcab9880-c2ba-11eb-858a-adcb153154d4.png)  


**FSRCNN의 큰 특징 3가지**
- 첫째, SRCNN이 Bicubic Interpolation으로 upscale을 먼저 한 후 convolution layer에 집어넣는 방식을 사용한 반면, FSRCNN은 LR 이미지를 그대로 convolution layer에 집어넣는 방식을 사용하여 convolution에서의 연산량을 줄였음
- 둘째, network 후반부에서 feature map의 width, height size를 키워주는 Deconvolution(transposed convolution)연산을 사용하여 HR 이미지를 만들었음.
- 셋째, SRCNN의 non-linear mapping 단계를 shrinking, mapping, expanding 세 단계로 분리하였음. 

그 결과 SRCNN에 비해 굉장히 연산량이 줄어들어 저자는 거의 실시간에 준하는 성능을 보일 수 있음을 강조하였으며, 연산량이 줄어든 만큼 convolution layer의 개수도 늘려주면서 정확도(PSNR)도 챙길 수 있음을 보여주었음.



![0_jPCy664hkmSJiVkP](https://user-images.githubusercontent.com/72849922/120270299-55a2d980-c2e4-11eb-9dc6-0c3a1298cc6d.png)

FSRCNN은 상대적으로 얕은 네트워크를 가지고 있어 각 구성 요소의 효과를 더 쉽게 배울 수 있으며,

FSRCNN-s 모델은 FSRCNN을 경량화시킨 모델로, 미미한 차이의 성능 저하를 감수하고 속도를 더욱 향상시켰으며 이는 SRCNN 보다 훨씬 빠른 속도를 보임

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
필자는 FSRCNN 모델을 경량화 시킨 FSRCNN-s 모델과 Optimal-FSRCNN 모델을 참고하여 FPGA에서 작동 가능한 초경량화된 FSRCNN-x 모델을 제안함

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


## Prepare
- train_x2.h5
```bash
python prepare.py --images_dir "./train_file" \
               --output_path "./output_path/train_x2.h5" \
               --scale 2 \
```
- eval_x2.h5
```bash
python prepare.py --images_dir "./eval_file" \
               --output_path "./output_path/eval_x2.h5" \
               --scale 2 \
               --eval
```


## Train
```bash
python train.py --train-file "./output_path/train_x2.h5" \
                --eval-file "./output_path/eval_x2.h5" \
                --outputs-dir "./outputs_dir" \
                --scale 2 \
                --lr 1e-3 \
                --batch-size 128 \
                --num-epochs 20 \
                --num-workers 8 \
                --seed 123                
```

## Test
```bash
python test.py --weights-file "outputs_dir/x2/x2_best.pth" \
               --image-file "data/butterfly_GT.bmp" \
               --scale 2
```


## Result
<table>
    <tr>
        <td><center>ORIGIN</center></td>
        <td><center>BICUBIC</center></td>
        <td><center>FSRCNN-s x2</center></td>
        <td><center>FSRCNN-x x2</center></td>
    </tr>
    <tr>
     <td>
    		<center><img src="./data/butterfly_GT.bmp"></center>
    	</td>
    	<td>
    		<center><img src="./data/butterfly_GT_new_bicubic_x2.bmp"></center>
    	</td>
    	<td>
    		<center><img src="./data/butterfly_GT_fsrcnn-s_x2.bmp"></center>
    	</td>
    	<td>
    		<center><img src="./data/butterfly_GT_fsrcnn-x_x2.bmp"></center>
    	</td>
    </tr>
</table>




| 유형 | 링크 |
|---|:---:|
| 참고 논문 | [Deep Learning-based Real-Time Super-Resolution Architecture Design](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002699585) |
| 참고 논문 |[Accelerating the Super-Resolution Convolutional Neural Network](https://github.com/KHS0616/SuperResolution/blob/master/Paper/FSRCNN.md)| 

