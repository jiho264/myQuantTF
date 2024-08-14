
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

