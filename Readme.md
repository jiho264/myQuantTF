
# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  



# ...
- torch base : 81.072%

여기 activation 8비트 실험결과에
softmax*v 하는 부분에 오류있어서 적용 안됨. 다 다시 실험해야함.

| Case Name                             | Weight   | Activation   | Attention | LayerNorm | Acc @ 1 |
| ------------------------------------- | -------- | ------------ | --------- | --------- | ------- |
| Base                                  | 32       | 32           | 32        | 32        | 81.068% |
| [W] MinMax_ch                         | 8        | 32           | 32        | 32        |         |
| [W] AdaRoundMinMax_ch                 | 4        | 32           | 32        | 32        |         |
| [W] MinMax_ch                         | 4        | 32           | 32        | 32        |         |
| [W] AdaRoundMinMax_ch + [A] MovingAvg | 4        | 8            | 32        | 32        |         |
| [W] MinMax_ch + [A] MovingAvg         | 4        | 8            | 32        | 32        | 79.952% |
| [W] MinMax_ch + [A] MovingAvg         | 4        | 8 + INT GELU | 32        | 32        | 79.514% |
| ALL per-ch + INT32에 softmax          | MinMax 4 | Dynamic 8    | Dynamic 8 | Dynamic 8 |         |
| ALL per-ch + INT8에 softmax           | MinMax 4 | Dynamic 8    | Dynamic 8 | Dynamic 8 |         |

- Dynamic : Dynamic MinMax Quantizer
- The final output (prediction) does not apply activation quantization.