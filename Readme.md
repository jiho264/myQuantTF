
# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  



# ...
- torch base : 81.072%

|    Case Name    |  Weight  | Activation | Attention | LayerNorm | Acc @ 1 | Test Time |
| :-------------: | :------: | :--------: | :-------: | :-------: | :-----: | :-------: |
|      Base       |    32    |     32     |    32     |    32     | 81.072% |   1:55    |
| ALL per-channel | MinMax 4 | Dynamic 8  | Dynamic 8 | Dynamic 8 | 79.426% |   5:17    |

- Dynamic : Dynamic MinMax Quantizer