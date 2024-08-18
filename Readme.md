
# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  



# 
- torch base : 81.072%
- my quantization implementation for torch ViT-B model 

| Case Name  | W   | A   | SoftMax | GELU | LN  | Acc @ 1 |
| ---------- | --- | --- | ------- | ---- | --- | ------- |
| Base       | 32  | 32  | 32      | 32   | 32  | 81.068% |
| [W] MinMax | 8   | 32  | 32      | 32   | 32  | 81.058% |
| [W] MinMax | 4   | 32  | 32      | 32   | 32  | 80.144% |



# Idea
- Weight에 AdaRound를 적용하기. Symmetric scheme에 기반하는데, 라운딩 벨류가 반 올림을 결정해준다고하면, INT GEMM 수식에 v는 영향을 주지 않음.
- 모든 Activation scaler는 Symmetric scheme에 기반하는데, 모델의 prediction이 최대한 덜 달라지게하는 step size를 선정해야함.
- INT GELU, INT SOftmax, INT LayerNorm 등을 적용하여 완전한 INT연산만 이루어지도록 하기



- cls_token, pos_embedding에 대한 결정. (FP32임)
  - I-ViT에서는 LN간 input과 output은 32bit로 표현됨.
  - class token과 position embedding값은 학습된 아주 작은 숫자들임.
  - 이를 8비트로하면 약 0.9%의 정보손실 있음.
  - 굳이 8비트까지 줄일 이유 매우 적음. 
  - 32비트로하면 다음 행렬곱에서 오버플로우 가능성 있음.
  - 즉 I-ViT와 동일하게 16비트로 변환하기.
  - 첫 LN의 input은 16비트가 됨. 이외 모든 LN의 in/out은 32bit
  - INT32로 표현된 INT16의 값임.
  - 뒤에서 INT32연산 하는데 여기서 굳이 8비트까지 써서 정확도 크리티컬하게 낮출 필요 없음.
  
- INT32 범위에서 quantized된 것과, INT8값이 INT32 누산기에서 존재하는 것은 엄연히 다름.
  - 이점 매우 유의해야함.
  - 어차피 밑의 LN에서도 INT32를 쓴다고 하는데, 절대 INT32로 Incoding된 값이 아님. 
  - 자료형만 INT32일 뿐, INT8간 행렬곱 결과가 