# LLAMA_Post_RandomProcess

离线 logits 驱动的采样后处理（C 实现）：`penalty -> temperature -> top-k -> top-p -> min-p -> sample`，并提供 Python golden 校验。

## Build

```bash
make
```

## Run

输出 16 个基于离线 logits 的采样 token（默认）：

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

运行旧版“固定 token 表随机 demo”：

```bash
./random_token_demo 16 --table
```

## Golden Verify

```bash
python3 tools/golden_verify.py ./random_token_demo 32
```

## Test

```bash
make test
```
