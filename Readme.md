# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  

# Results
- torch base : 81.072%
- my quantization implementation for torch ViT-B model 

| [W]                | [A]    | W   | A   | SoftMax | GELU  | LN    | IdAdd | Acc @ 1 |
| ------------------ | ------ | --- | --- | ------- | ----- | ----- | ----- | ------- |
| Base               | Base   | 32  | 32  | FP      | FP    | FP    | FP    | 81.068% |
| [W]Abs             | -      | 8   | 32  | FP      | FP    | FP    | FP    | 81.074% |
| [W]Abs             | -      | 4   | 32  | FP      | FP    | FP    | FP    | 79.794% |
| [W]Abs             | [A]Mov | 8   | 8   | FP      | FP    | FP    | FP    | 78.406% |
| [W]Abs             | [A]Mov | 4   | 8   | FP      | FP    | FP    | FP    | 76.894% |
| [W]Abs             | [A]Mov | 8   | 8   | I-ViT   | I-ViT | I-ViT | 16    | 77.064% |
| [W]Abs             | [A]Mov | 4   | 8   | I-ViT   | I-ViT | I-ViT | 16    | 72.964% |
| [W]AdaRound(Layer) | [A]Mov | 4   | 8   | I-ViT   | I-ViT | I-ViT | 16    | 79.076% |
| [W]AdaRound(Block) | [A]Mov | 4   | 8   | I-ViT   | I-ViT | I-ViT | 16    |         |
| [W]Abs             | [A]Mov | 4   | 4   | I-ViT   | I-ViT | I-ViT | 16    | 0.134%  |


- Weight Quantizer : Absolute Max Quantization
- Activation Quantizer : Moving Average Absolute Max Quantization (momentum 0.95, 2048 images)
- Softmax : INT8 input -> calculate with INT16 -> requantize to UINT8
- LayerNorm : INT16 input -> calulate with INT16 -> requantize to INT8
- AdaRound : [PerLayer, PerBlock (MSA, MLP), PerEncoder, PerModel]
  - Batch : 32
  - Calib set for adaround : 1024 non-labeled images from same domain with training set.
  - LR for rounding value : 1e-2
  - LR for LSQ : 4e-5
  - length of Iteration : 10k
  - Do early stopping when the regularization loss is small than 1e-5.
- LSQ : Learn stepsize quantization for activation.
  - Esser, Steven K., et al. "Learned step size quantization." ICLR 2020.
  - Use the BRECQ's LSQ implementation.

# Todo
- [ ] Find an optimal LR for rounding value computing.
- [ ] Find the optimal granularity in the quantized model.
- [ ] FIXME input quant and last ln act of encoder dose NOT CONSIDERED.

# Idea
- Weight에 AdaRound를 적용하기. Symmetric scheme에 기반하는데, 라운딩 벨류가 반 올림을 결정해준다고하면, INT GEMM 수식에 v는 영향을 주지 않음. -> 좋으면 좋았지 악영향이 있을 수가 없음. 
  - LR of V 1e-3 ~ 1e-4로 하면 처음에 너무 안 튀어올라서, reg term 계산하고 바로 사그라든 뒤 원래 residual과 같은 값으로 계산됨. 1e-2로 변경하고 실험 중. 수 천 iter에 걸쳐 학습되는 양상을 보임.
- 모든 Activation scaler는 Symmetric scheme에 기반하는데, 모델의 prediction이 최대한 덜 달라지게하는 step size를 선정해야함.
- 모든 Quantization에서 Symmetric scheme을 이용해야 하는 이유 : zeropoint있으면 INT로 변환시 INT8 범위에서 표현 못 할 수도 있음.
- INT GELU, INT SOftmax, INT LayerNorm 등을 적용하여 완전한 INT연산만 이루어지도록 하기
- LN의 input은 모두 16비트 해상도의 INT32 값들임. 여기를 8비트로 낮추면 완전히 망가져버림. 그래도 다행인건, LN은 행렬곱보다 부담이 덜 한 선형연산이라는 점.


# Memo
- seed 고정 안 돼있었는데 왜 항상 동일 정확도나왔는진 잘 모르겠음. 메번 데이터로더 달랐는데 지금까지 moving avg act quantizer가 모든 s_a 동일하게 구하고있었음. >> 이동평균 MinMax가 아주 균일한 결과를 나타냈다고 볼 수 있음. (128 * 16 images)
