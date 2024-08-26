# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  

# Results
- torch base : 81.072%
- my quantization implementation for torch ViT-B model 

| [W]              | [A]    | W   | A   | SoftMax | GELU    | LN      | IdAdd | Acc @ 1 |
| ---------------- | ------ | --- | --- | ------- | ------- | ------- | ----- | ------- |
| Base             | Base   | 32  | 32  | FP      | FP      | FP      | FP    | 81.068% |
| [W]Abs           | -      | 8   | 32  | FP      | FP      | FP      | FP    | 81.074% |
| [W]Abs           | -      | 4   | 32  | FP      | FP      | FP      | FP    | 79.794% |
| [W]Abs           | [A]Mov | 8   | 8   | FP      | FP      | FP      | FP    | 78.406% |
| [W]Abs           | [A]Mov | 8   | 8   | I-ViT-8 | I-ViT-8 | I-ViT-8 | 16    | 77.064% |
| [W]AdaR(Layer)   | [A]Mov | 8   | 8   | I-ViT-8 | I-ViT-8 | I-ViT-8 | 16    | 77.772% |
| [W]AdaR(Block)   | [A]Mov | 8   | 8   | I-ViT-8 | I-ViT-8 | I-ViT-8 | 16    | 77.674% |
| [W]AdaR(Encoder) | [A]Mov | 8   | 8   | I-ViT-8 | I-ViT-8 | I-ViT-8 | 16    | 78.782% |
| [W]Abs           | [A]Mov | 4   | 8   | FP      | FP      | FP      | FP    | 76.894% |
| [W]Abs           | [A]Mov | 4   | 8   | I-ViT-8 | I-ViT-8 | I-ViT-8 | 16    | 72.964% |
| [W]AdaR(Layer)   | [A]Mov | 4   | 8   | I-ViT-8 | I-ViT-8 | I-ViT-8 | 16    | 79.076% |
| [W]AdaR(Block)   | [A]Mov | 4   | 8   | I-ViT-8 | I-ViT-8 | I-ViT-8 | 16    | 78.484% |
| [W]AdaR(Encoder) | [A]Mov | 4   | 8   | I-ViT-8 | I-ViT-8 | I-ViT-8 | 16    | 78.782% |
| [W]Abs           | [A]Mov | 4   | 4   | I-ViT-8 | I-ViT-8 | I-ViT-8 | 16    | 0.134%  |


## Table 1 from RepQuant 
- Li, Zhikai, et al. "RepQuant: Towards Accurate Post-Training Quantization of Large Transformer Models via Scale Reparameterization." arXiv preprint arXiv:2402.05628 (2024).
- Zhikai Li : the Author of I-ViT(ICCV 2023), RepQ-ViT(ICCV 2023), ReqQuant(24.02 arXiv)

| Method          | Prec. (W/A) | ViT-S | ViT-B | DeiT-T | DeiT-S | DeiT-B | Swin-S | Swin-B |
| --------------- | ----------- | ----- | ----- | ------ | ------ | ------ | ------ | ------ |
| Full-Precision  | 32/32       | 81.39 | 84.54 | 72.21  | 79.85  | 81.80  | 83.23  | 85.27  |
| --------------- | ----------- | ----- | ----- | ------ | ------ | ------ | ------ | ------ |
| FQ-ViT          | 4/4         | 0.10  | 0.10  | 0.10   | 0.10   | 0.10   | 0.10   | 0.10   |
| PTQ4ViT         | 4/4         | 42.57 | 30.69 | 36.96  | 34.08  | 64.39  | 76.09  | 74.02  |
| APQ-ViT         | 4/4         | 47.95 | 41.41 | 47.94  | 43.55  | 67.48  | 77.15  | 76.48  |
| RepQ-ViT        | 4/4         | 65.05 | 68.48 | 57.43  | 69.03  | 75.61  | 79.45  | 78.32  |
| RepQuant        | 4/4         | 73.28 | 77.84 | 64.44  | 75.21  | 78.46  | 81.52  | 82.80  |
| --------------- | ----------- | ----- | ----- | ------ | ------ | ------ | ------ | ------ |
| FQ-ViT          | 6/6         | 4.26  | 0.10  | 58.66  | 45.51  | 64.63  | 66.50  | 52.09  |
| PSAQ-ViT        | 6/6         | 37.19 | 41.52 | 57.58  | 63.61  | 67.95  | 72.86  | 76.44  |
| Ranking         | 6/6         | -     | 75.26 | -      | 74.58  | 77.02  | -      | -      |
| PTQ4ViT         | 6/6         | 78.63 | 81.65 | 69.68  | 76.28  | 80.25  | 82.38  | 84.01  |
| APQ-ViT         | 6/6         | 79.10 | 82.21 | 70.49  | 77.76  | 80.42  | 82.67  | 84.18  |
| NoisyQuant      | 6/6         | 78.65 | 82.32 | -      | 77.43  | 80.70  | 82.86  | 84.68  |
| RepQ-ViT        | 6/6         | 80.43 | 83.62 | 70.76  | 78.90  | 81.27  | 82.79  | 84.57  |
| RepQuant        | 6/6         | 80.51 | 83.75 | 70.89  | 79.06  | 81.41  | 82.93  | 84.86  |
| --------------- | ----------- | ----- | ----- | ------ | ------ | ------ | ------ | ------ |
| [QAT] I-ViT     | 8/8/8       | 81.27 | 84.76 | 72.24  | 80.12  | 81.74  | 81.50  | 83.01  |

## Implementation details
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
