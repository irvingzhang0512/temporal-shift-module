# Jetbot模型移植

## 0. 前言
+ 总目标：实现手势识别模型移植到Jetbot上。
+ Features
  + [x] 各类模型转换为TVM Relay
  + [x] Auto Tuning
  + [ ] 利用交叉编译测试 Auto Tuning 的结果
  + [ ] 在Jetbot本地测试模型
+ 存在的问题：
  + [ ] 使用CUDA进行Auto Tuning会失败。好像不是Jetbot本身的问题，因为在服务器上执行Auto Tuning也会失败。