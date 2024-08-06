
# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  



# ...
- torch base : 81.072%

| Case Name                     | Weight   | Activation | Attention       | LayerNorm | Acc @ 1 |
| ----------------------------- | -------- | ---------- | --------------- | --------- | ------- |
| Base                          | 32       | 32         | 32              | 32        | 81.068% |
| [W] MinMax_ch                 | 8        | 32         | 32              | 32        | 81.058% |
| [W] MinMax_ch                 | 4        | 32         | 32              | 32        | 80.144% |
| [W] MinMax_ch + [A] MovingAvg | 4        | 8          | 32 (gamma 3e-4) | 32        | 80.030% |
| [W] MinMax_ch + [A] MovingAvg | 4        | 4          | 32 (gamma 3e-4) | 32        | 1.758%  |
| [W] MinMax_ch + [A] MovingAvg | 4        | 8          | 32 (gamma 1e-4) | 32        | 80.020% |
| [W] MinMax_ch + [A] MovingAvg | 4        | 8          | 32              | 32        | 79.976% |
| ALL per-ch + INT32에 softmax  | MinMax 4 | Dynamic 8  | Dynamic 8       | Dynamic 8 | 79.534% |
| ALL per-ch + INT8에 softmax   | MinMax 4 | Dynamic 8  | Dynamic 8       | Dynamic 8 | 79.426% |

- Dynamic : Dynamic MinMax Quantizer