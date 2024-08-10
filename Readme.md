
# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  



# ...
- torch base : 81.072%


| Case Name                       | W   | A   | Attn | GELU   | LN  | Acc @ 1 |
| ------------------------------- | --- | --- | ---- | ------ | --- | ------- |
| Base                            | 32  | 32  | 32   | 32     | 32  | 81.068% |
| [W] MinMax                      | 8   | 32  | 32   | 32     | 32  | 81.058% |
| [W] MinMax                      | 4   | 32  | 32   | 32     | 32  | 80.144% |
| [W] AdaRoundMinMax + [A] MovAvg | 4   | 8   | 32   | 32     | 32  | 80.502% |
| [W] MinMax + [A] MovAvg         | 4   | 8   | 32   | 32     | 32  | 79.952% |
| [W] MinMax + [A] MovAvg         | 4   | 8   | 32   | I-BERT | 32  | 79.514% |


- Dynamic : Dynamic MinMax Quantizer
- All weight scaler : per channel
- All activation scaler : per tensor
- The final output (prediction) does not apply activation quantization.
- Attn : the computation process for softmax(Q@K/sqrt(d_k))