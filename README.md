# LLAMA_Post_RandomProcess

固定离线 token 阵列的随机 token 获取逻辑（C 实现），并提供 Python golden 校验。

## Build

```bash
make
```

## Run

输出 16 个随机 token（默认）：

```bash
./random_token_demo
```

指定输出数量：

```bash
./random_token_demo 32
```

## Golden Verify

```bash
python3 /home/runner/work/LLAMA_Post_RandomProcess/LLAMA_Post_RandomProcess/tools/golden_verify.py ./random_token_demo 32
```

## Test

```bash
make test
```
