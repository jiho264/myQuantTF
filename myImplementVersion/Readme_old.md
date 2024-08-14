

DEPRECATED at 2024.08.14

when i realized that the bulding int inference code is very difficult, so i decided to use the I-ViT's code.



# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  

[OLD version]

- my quantization implementation for torch ViT-B model 
- working with [myUtils] package




# myQuant.py
- torch base : 81.072%
- my quantization implementation for torch ViT-B model 

| Case Name                       | W   | A   | SoftMax  | GELU     | LN  | Acc @ 1 |
| ------------------------------- | --- | --- | -------- | -------- | --- | ------- |
| Base                            | 32  | 32  | 32       | 32       | 32  | 81.068% |
| [W] MinMax                      | 8   | 32  | 32       | 32       | 32  | 81.058% |
| [W] MinMax                      | 4   | 32  | 32       | 32       | 32  | 80.144% |
| [W] AdaRoundMinMax + [A] MovAvg | 4   | 8   | 32       | 32       | 32  | 80.502% |
| [W] MinMax + [A] MovAvg         | 4   | 8   | 32       | 32       | 32  | 79.952% |
| [W] MinMax + [A] MovAvg         | 4   | 8   | 32       | I-BERT 8 | 32  | 79.558% |
| [W] MinMax + [A] MovAvg         | 4   | 8   | I-BERT 8 | 32       | 32  | 00000%  |
| [W] MinMax + [A] MovAvg         | 4   | 8   | I-BERT 8 | I-BERT 8 | 32  | 00000%  |


- Dynamic : Dynamic MinMax Quantizer
- All weight scaler : per channel
- All activation scaler : per tensor
- The final output (prediction) does not apply activation quantization.
- Attn : the computation process for softmax(Q@K/sqrt(d_k))