# Jetbot模型移植

## 0. 前言
+ 总目标：实现手势识别模型移植到Jetbot上。
+ Features
  + [x] 将 `tsm/models` 下的各类模型转换为TVM Relay
    + 脚本：`to_relay_model.py`
  + [x] Auto Tuning，并导出部署所需的文件。
    + 脚本：`auto_tuning.py`。
  + [x] 利用交叉编译测试 Auto Tuning 的结果，在Jetbot本地测试模型。
    + 脚本：`jetbot_tvm_demo.py`
+ TODO
  + [ ] 测试对于一个Auto Tuning的结果，能否直接改变权重而不重新Auto Tuning。

## 1. 碰到的问题

### 1.1. 使用CUDA进行Auto Tuning会失败
+ 描述：好像不是Jetbot本身的问题，因为在服务器上执行Auto Tuning也会失败。
+ 解决：
  + 在X月X日git pull最新的代码后并编译后得到上述错误的情况，是TVM本身的问题。
  + 过了一两天，pull了最新的代码后就没有问题了。
  + 尝试了将代码还原到4.15日的版本，也没有问题了。

### 1.2. TVM在Jetbot上编译失败
+ 描述：老是在编译到80%+的时候说找不到某个`.h`文件，大概名字是`IR/XXX_AMDGPU.h`。
+ 解决：
  + 因为在Jetbot上的llvm是自己编译的，所以有一些头文件不存在。
  + llvm不要自己编译，而要通过 `sudo apt install llvm` 来安装。
  + 编译TVM源码时，设置 `config.cmake` 中的 `USE_CUDA ON` / `USE_LLVM ON`，然后就没问题了。

### 1.3. 交叉编译/远程AutoTuning时报错
+ 描述：交叉编译/远程AutoTuning老是出现奇怪的错误。
+ 解决：
  + 主要是一些设置问题，主要就是设置 `target/target_host`。
  + `target` 设置为 `tvm.target.cuda(model="nano")` 和 `'cuda'` 都可以运行。
  + `target_host` 必须设置为Jetbot上的形式，即 `llvm -target=aarch64-linux-gnu`。
  + 另外，需要设置GPU架构，Jetbot中设置 `set_cuda_target_arch('sm_53')`，该方法来自 `from tvm.autotvm.measure.measure_methods import set_cuda_target_arch`。