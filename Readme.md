
# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  



# 
- torch base : 81.072%
- my quantization implementation for torch ViT-B model 

| [W]    | [A]    | W   | A   | SoftMax | GELU  | LN    | IdAdd | Acc @ 1 |
| ------ | ------ | --- | --- | ------- | ----- | ----- | ----- | ------- |
| Base   | Base   | 32  | 32  | FP      | FP    | FP    | FP    | 81.068% |
| [W]Abs | -      | 8   | 32  | FP      | FP    | FP    | FP    | 81.074% |
| -      | [A]Mov | 32  | 8   | FP      | FP    | FP    | FP    | 78.994% |
| [W]Abs | [A]Mov | 8   | 8   | FP      | FP    | FP    | FP    | 78.474% |
| [W]Abs | -      | 4   | 32  | FP      | FP    | FP    | FP    | 79.794% |
| [W]Abs | [A]Mov | 4   | 8   | FP      | FP    | FP    | FP    | 76.874% |
| -      | [A]Mov | 32  | 8   | I-ViT   | I-ViT | I-ViT | 16    | 75.890% |
| [W]Abs | [A]Mov | 8   | 8   | I-ViT   | I-ViT | I-ViT | 16    | 77.054% |
| [W]Abs | [A]Mov | 4   | 8   | I-ViT   | I-ViT | I-ViT | 16    | 72.964% |

- Weight Quantizer : Absolute Max Quantization
- Activation Quantizer : Moving Average Absolute Max Quantization (momentum 0.95, 2048 images)
- Softmax : Calculated by INT16, quantized by UINT8




# Idea
- Weight에 AdaRound를 적용하기. Symmetric scheme에 기반하는데, 라운딩 벨류가 반 올림을 결정해준다고하면, INT GEMM 수식에 v는 영향을 주지 않음.
- 모든 Activation scaler는 Symmetric scheme에 기반하는데, 모델의 prediction이 최대한 덜 달라지게하는 step size를 선정해야함.
- INT GELU, INT SOftmax, INT LayerNorm 등을 적용하여 완전한 INT연산만 이루어지도록 하기
- LN의 input은 모두 16비트 해상도의 INT32 값들임. 여기를 8비트로 낮추면 완전히 망가져버림. 그래도 다행인건, LN은 행렬곱보다 부담이 덜 한 선형연산이라는 점.