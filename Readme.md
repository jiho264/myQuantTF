
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
| [W] AbsMax | 8   | 8   | 16 -> 7 | 8    | 8   | 77.190% |




# Idea
- Weight에 AdaRound를 적용하기. Symmetric scheme에 기반하는데, 라운딩 벨류가 반 올림을 결정해준다고하면, INT GEMM 수식에 v는 영향을 주지 않음.
- 모든 Activation scaler는 Symmetric scheme에 기반하는데, 모델의 prediction이 최대한 덜 달라지게하는 step size를 선정해야함.
- INT GELU, INT SOftmax, INT LayerNorm 등을 적용하여 완전한 INT연산만 이루어지도록 하기



- cls_token : 16
- pos embedding : 8
- LN : 8
- gelu : 8
- softmax : 16 -> quant to 8
- identity addition : 16