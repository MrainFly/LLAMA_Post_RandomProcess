# LLAMA_Post_RandomProcess

当前仓库已升级为 **llama.cpp 风格的采样后处理 demo**：默认模式使用离线 logits 驱动采样链，流程为：

`penalty -> top-k -> top-p -> min-p -> temperature -> sample`

另外仍保留旧版固定 token 表随机 demo（`--table`）用于回归对照。

## 仓库内容

- `src/sampler_llama.cpp`：llama 风格采样链（penalty / top-k / top-p / min-p / temperature / sampling）
- `src/random_token.c`：采样参数默认值、离线 logits 数据、C 桥接接口
- `src/main.c`：CLI 参数解析与 demo 入口
- `tools/golden_verify.py`：Python golden 校验
- `tools/llama_infer_tiny.py`：轻量“llama-like”推理版本（小体积、单文件、无第三方依赖）

## Build

```bash
make
```

## Run

默认输出 16 个基于离线 logits 的采样 token：

```bash
./random_token_demo
```

指定输出数量：

```bash
./random_token_demo 32
```

使用自定义采样参数（固定 seed 可复现）：

```bash
./random_token_demo 32 --seed 42 --top-k 4 --top-p 0.9 --min-p 0.05 --temperature 0.8
```

关闭 history penalty：

```bash
./random_token_demo 32 --history-window 0
```

运行旧版固定 token 表随机 demo：

```bash
./random_token_demo 16 --table
```

## Tiny Inference（轻量推理版本）

如果你希望在本仓库内直接看到“推理 + 采样”的完整链路，可运行：

```bash
python3 tools/llama_infer_tiny.py --tokens 128000 128001 128002 --steps 6
```

该脚本是 **小体积推理原型**（非完整 llama.cpp 内核），用于把“生成 logits”的步骤也放进仓库，便于联调采样流程。

## Golden Verify

```bash
python3 tools/golden_verify.py ./random_token_demo 32
```

## Test

```bash
make test
```
