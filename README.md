### agent policy版本:
v2:有減掉上一回合的狀態，但沒有考慮敵人的情況
v3:有減掉上一回合的狀態，也有考慮敵人的情況

### train的方式:
| | v2 30萬*2| v2 10萬*6 | v3 30萬*2|v3 10萬*6|
| :----: | :----:  | :----: |:----:|:----:|
| dqn| v|   v| v| v|
| ppo |  v |  v | v| v|
共八組


