
# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  



# ...
- torch base : 81.072%

| Case Name                             | Weight   | Activation   | Attention | LayerNorm | Acc @ 1 |
| ------------------------------------- | -------- | ------------ | --------- | --------- | ------- |
| Base                                  | 32       | 32           | 32        | 32        | 81.068% |
| [W] MinMax_ch                         | 8        | 32           | 32        | 32        | 81.058% |
| [W] AdaRoundMinMax_ch                 | 4        | 32           | 32        | 32        | 80.668% |
| [W] MinMax_ch                         | 4        | 32           | 32        | 32        | 80.144% |
| [W] AdaRoundMinMax_ch + [A] MovingAvg | 4        | 8            | 32        | 32        | 80.536% |
| [W] MinMax_ch + [A] MovingAvg         | 4        | 8            | 32        | 32        | 79.996% |
| [W] MinMax_ch + [A] MovingAvg         | 4        | 8 + INT GELU | 32        | 32        | 79.568% |
| ALL per-ch + INT32에 softmax          | MinMax 4 | Dynamic 8    | Dynamic 8 | Dynamic 8 | 79.534% |
| ALL per-ch + INT8에 softmax           | MinMax 4 | Dynamic 8    | Dynamic 8 | Dynamic 8 | 79.426% |

- Dynamic : Dynamic MinMax Quantizer
- The final output (prediction) does not apply activation quantization.