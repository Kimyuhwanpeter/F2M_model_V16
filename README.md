# F2M_model_V16
* 저주파 및 고주파를 이용한 loss
* SAM 이용 (현재 서브컴으로 실험 중)
* 모델 입력에 쓰인 sigmoid를 변형 (중간 sigmoid는 고정), 1/(1+exp(-4x))
* Generator 마지막 단에 1x1 conv 추가
* 오직 Adam만 이용 (현재 코랩에서 실험 중)

## 실험 1 (서브컴)
* 입력에서 생성된 영상을 후처리 한 결과 (나이 성능은 원본 및 후처리 결과를 따로 측정해봐야 함)
![f1](https://github.com/Kimyuhwanpeter/F2M_model_V16/blob/main/Figure_1.png)
<br/>

* Sharpening filter는 아래와 같이 사용
![f2](https://github.com/Kimyuhwanpeter/F2M_model_V16/blob/main/Figure_2.png)
