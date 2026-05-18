# LLAMA 随机采样与轻量推理说明（已更新）

## 1. 当前状态（与旧文档差异）

仓库当前默认路径已不是“固定表 + 取模随机”。

默认 `./random_token_demo` 走的是：

- 离线 logits 输入
- llama 风格 sampling chain
- 固定 seed 的可复现采样

旧版固定 token 表随机流程仍保留为兼容模式：

- `./random_token_demo --table`

---

## 2. 当前实现结构

### 2.1 采样链（核心）

`src/sampler_llama.cpp` 中按顺序执行：

1. history penalties（repeat / frequency / presence）
2. top-k
3. top-p
4. min-p
5. temperature
6. 最终概率采样（temperature=0 时走贪心）

并在采样后更新 history。

### 2.2 参数与数据

`src/random_token.c` 负责：

- 默认采样参数
- sampler state 初始化
- 离线 logits 数据与 token id 数据
- C/C++ 桥接调用

### 2.3 CLI

`src/main.c` 支持：

- `--seed`
- `--temperature`
- `--top-k`
- `--top-p`
- `--min-p`
- `--repeat-penalty`
- `--frequency-penalty`
- `--presence-penalty`
- `--history-window`
- `--table`（兼容旧 demo）

---

## 3. 校验方式

Python golden：

- `tools/golden_verify.py`

单元测试：

- `make test`

目标是校验固定参数下 C 输出序列的稳定性与可回归性。

---

## 4. 轻量推理版本（已拉入仓库）

为满足“推理版本也在仓库内、且体积尽量小”的需求，新增：

- `tools/llama_infer_tiny.py`

它是一个 **tiny llama-like inference 原型**，用于提供可复现 logits，
再把 logits 送入采样流程进行联调。该脚本特性：

- 单文件
- 无第三方依赖
- 可直接运行
- 适合调通“推理 -> 采样”链路

注意：它不是完整 llama.cpp 推理内核，不替代官方实现。

---

## 5. 与 llama.cpp 的对齐边界

当前仓库“采样后处理思路”与 llama.cpp 接近；
“完整模型推理内核”仍以官方项目为准。

因此本仓库定位是：

- 轻量、可复现、可测试的采样实验仓库
- 带一个小体积推理原型，便于端到端调试

