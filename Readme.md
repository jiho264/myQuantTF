
# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  



# ...
- torch base : 81.072%

| Case Name                    | Weight   | Activation | Attention | LayerNorm | Acc @ 1 | Time for test |
| ---------------------------- | -------- | ---------- | --------- | --------- | ------- | ------------- |
| Base                         | 32       | 32         | 32        | 32        | 81.072% | 1:55          |
| Weight MinMax per-ch         | 8        | 32         | 32        | 32        | 81.058% | 2:08          |
| Weight MinMax per-ch         | 4        | 32         | 32        | 32        | 80.144% | 2:06          |
| ALL per-ch + INT32에 softmax | MinMax 4 | Dynamic 8  | Dynamic 8 | Dynamic 8 | 79.534% | 4:45          |
| ALL per-ch + INT8에 softmax  | MinMax 4 | Dynamic 8  | Dynamic 8 | Dynamic 8 | 79.426% | 5:17          |

- Dynamic : Dynamic MinMax Quantizer