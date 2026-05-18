# LLAMA 随机采样思路说明

## 1. 背景

这个仓库当前实现的是一个**离线随机 token demo**，它能验证“固定种子 + 固定候选表 + 固定随机过程”这一件事，但它**还不是 llama 官方内核的真实做法**。

如果整个工程后续要真正依赖 llama 官方思路，那么核心原则应该是：

- **不要自己发明一套随机规则**
- **要围绕 llama 的 sampler / sampling chain 来组织流程**
- **随机过程的输入应该是 logits 或候选分布，而不是先写死 token 表后直接抽签**

下面把 llama 官方思路、当前仓库思路、以及 Python 侧应该怎么落地，统一梳理一下。

---

## 2. llama 官方在这块的思路

这里参考的是 `llama.cpp` 当前公开实现里的 sampling 设计，核心不在“随机一个 token id”，而在“**对模型输出分布做一串约束和变换后，再从结果分布里采样**”。

### 2.1 官方流程的核心对象是候选分布，不是固定 token 表

llama 内核做推理时，模型先给出当前 step 的 **logits**。  
这些 logits 覆盖的是整个词表，每个 token 都有一个分数。

后处理阶段并不是直接：

1. 准备一个固定 token 数组
2. 跑一个随机数
3. 对数组取模

而是：

1. 读取当前 step 的 logits
2. 构造成候选 token 集合
3. 依次应用 sampler chain
4. 最后再根据剩余分布采样出 1 个 token

### 2.2 官方是“sampler chain”串联处理

在 `llama.cpp` 的公共 sampling 初始化逻辑里，会构建一条 sampler chain。  
这条链不是单一算法，而是一组按顺序执行的约束器 / 变换器，例如：

- logit bias
- repetition / frequency / presence penalties
- top-k
- top-p
- min-p
- typical
- temperature
- mirostat
- grammar 约束
- reasoning budget 相关约束
- 最后的 dist sampler

它的本质是：

- 前面的 sampler 负责**裁剪或重排候选分布**
- 最后一个 sampler 负责**真正选出 token**

### 2.3 官方随机过程依赖 seed，但 seed 只是采样器的一部分

官方实现里也有 seed，但 seed 的作用不是“驱动一个独立的自定义随机 token 系统”，而是：

- 初始化 sampler 的随机状态
- 在已经被过滤过的候选分布上做可复现采样

也就是说，**seed 只负责复现性，不负责定义采样规则本身**。  
采样规则本身来自 sampler chain。

### 2.4 官方随机采样前会先做分布约束

以常见路径来说，流程通常接近下面这个思路：

1. 从模型拿到 logits
2. 根据历史 token 做 penalty
3. 应用 logit bias
4. 根据 grammar 或格式约束屏蔽非法 token
5. 应用 temperature
6. 应用 top-k / top-p / min-p 等裁剪
7. 对保留下来的候选计算概率
8. 用带 seed 的随机采样器从概率分布中抽取一个 token
9. 把这个 token 再反馈回 sampler 状态，供下一轮使用

所以官方思想是：

> **随机不是单独存在的功能，而是 logits 后处理链上的最后一步。**

### 2.5 官方思路和这个仓库当前 demo 的最大差异

官方是在“**全词表分布**”上工作；  
当前仓库是在“**固定离线 token 表**”上工作。

官方是在“**采样前先做约束与过滤**”；  
当前仓库是“**直接线性同余随机后取模**”。

官方是“**随机过程依赖模型输出**”；  
当前仓库是“**随机过程完全脱离模型输出**”。

---

## 3. 当前仓库的思路

当前实现集中在：

- `src/random_token.c`
- `tools/golden_verify.py`

### 3.1 当前 C 实现实际做了什么

当前 C 代码的逻辑非常简单：

1. 定义固定全局种子 `0x12345678`
2. 定义固定离线 token 表 `128000 ~ 128015`
3. 每次调用时用线性同余公式推进状态
4. 用 `state % count` 选择 token

这个逻辑的优点是：

- 简单
- 可复现
- 容易做 golden verify

但它本质上只是一个**固定候选表随机游走 demo**。

### 3.2 当前实现为什么不等价于 llama 官方采样

因为它缺少官方采样最关键的几个输入和步骤：

- 没有 logits
- 没有全词表候选分数
- 没有 penalty
- 没有 temperature
- 没有 top-k / top-p / min-p
- 没有 grammar 约束
- 没有历史 token 驱动的状态更新

所以当前实现只能表达：

> “我有一小组离线 token，要按固定伪随机顺序取值”

而不能表达：

> “我正在复现 llama 的 token sampling 逻辑”

### 3.3 当前 Python 校验脚本在验证什么

`golden_verify.py` 并不是在验证 llama 官方采样，  
它验证的是：

- Python 中的线性同余状态推进
- 是否与 C 代码完全一致

所以它现在是一个**当前 demo 的 golden**，不是 **llama 官方 sampling 的 golden**。

---

## 4. 如果要按 llama 官方思路改，应该怎样理解目标

建议把目标拆成两层：

### 4.1 第一层：接口目标

工程不要再把“随机 token”定义成：

- 输入：无
- 输出：从固定数组里抽一个 token

而应该定义成：

- 输入：当前 step 的 logits / candidate scores
- 输入：采样参数（seed、temperature、top-k、top-p、min-p 等）
- 输入：历史 token
- 输入：可选约束（grammar / bias / penalty）
- 输出：被采样出的 token

### 4.2 第二层：实现目标

实现上不要把“随机模块”写成独立逻辑块，  
而要写成“**sampling pipeline 的最后一段**”。

也就是说，模块边界应该是：

1. 上游负责提供 logits
2. sampling 模块负责按 llama 规则处理候选分布
3. sampling 模块返回 token，并更新内部状态

---

## 5. Python 代码的实现思路

如果后续要先用 Python 版本把官方思路跑通，建议按下面方式组织。

## 5.1 先明确 Python 版本的目标：不是复刻当前 demo，而是复刻 sampling pipeline

Python 版本不应该只是把下面逻辑搬过去：

- 固定 token 表
- LCG
- `% len(table)`

更合理的目标是：

- 输入 logits
- 输出按 llama 风格采样后的 token

### 5.2 建议的 Python 模块职责

可以拆成下面几个概念层：

#### A. `SamplingConfig`

负责保存采样参数，例如：

- `seed`
- `temperature`
- `top_k`
- `top_p`
- `min_p`
- `repeat_penalty`
- `frequency_penalty`
- `presence_penalty`
- `grammar_enabled`

这个对象的作用是把“参数定义”从“采样执行”里分离出来。

#### B. `SamplerState`

负责保存运行态数据，例如：

- RNG 状态
- 历史 token
- 上一轮采样结果

这部分很重要，因为 llama 官方采样不是纯函数，它和历史上下文有关。

#### C. `CandidateBuilder`

负责把 logits 转成候选列表。  
每个候选项至少应该包含：

- `token_id`
- `logit`
- `prob`（在 softmax 之后得到）

#### D. `SamplerChain`

负责按顺序执行各个 sampler：

1. logit bias
2. penalties
3. grammar mask
4. temperature
5. top-k
6. top-p
7. min-p
8. 最终随机采样

这个对象就是 Python 版里最接近 llama 官方思想的部分。

### 5.3 Python 侧推荐的执行顺序

每一轮采样可以按这个顺序：

1. 接收一组 logits
2. 构造 candidate 列表
3. 根据历史 token 做 repeat / frequency / presence penalty
4. 应用 logit bias
5. 如果有 grammar 规则，屏蔽非法 token
6. 应用 temperature
7. 做 top-k 裁剪
8. 做 top-p 裁剪
9. 做 min-p 裁剪
10. 对剩余候选归一化为概率分布
11. 用固定 seed 对应的 RNG 采样一个 token
12. 把采样结果写回 history

这才是“接近 llama 官方采样思路”的 Python 实现方向。

### 5.4 Python 侧最关键的注意点

#### 注意点 1：不要先固定一张离线 token 表再随机

如果先把候选空间缩成一个人工维护的小数组，那么本质就已经偏离官方逻辑了。  
正确做法应该是让“候选保留规则”来自 sampler chain，而不是来自手写表。

#### 注意点 2：随机动作要发生在概率分布上

不能只做：

- `index = rng % N`

应该做：

- 先基于 logits 得到概率
- 再基于概率抽样

#### 注意点 3：要让历史 token 参与后处理

如果没有历史 token，很多 llama 官方采样里的 penalty 都无法落地。  
所以 Python 实现至少要维护最近若干个 token。

#### 注意点 4：golden 的定义也要升级

如果 Python 将来改成官方 sampling 风格，那么 golden 也不能再只比对 LCG 输出。  
更合理的 golden 应该是：

- 给定固定 logits
- 给定固定采样参数
- 给定固定 seed
- 验证 Python 与目标实现是否输出相同 token 序列

---

## 6. 对当前工程的落地建议

如果这个仓库后续确实要向 llama 官方方案靠拢，建议按下面顺序推进：

### 阶段 1：先把“当前 demo”重新命名为简化版

先在认知上明确：

- 当前 `random_token.c` 是简化版演示
- 它不是 llama 官方 sampling 的复现

这一步很重要，否则后面讨论容易混淆。

### 阶段 2：把输入从“固定 token 表”升级为“候选分数”

即使暂时没有真实 llama logits，也应该先把接口改成可接收候选分数。  
这样后续才能无缝切到真实 llama 输出。

### 阶段 3：先在 Python 里做一版 sampler chain 原型

因为 Python 更适合验证流程，建议优先在 Python 中把这些规则跑通：

- temperature
- top-k
- top-p
- min-p
- penalty
- seed 可复现

等 Python 行为稳定后，再决定是否映射回 C 实现。

### 阶段 4：把 golden 从“LCG golden”升级为“sampling golden”

未来 golden 的重点应该从：

- “固定表 + 固定取模”

升级成：

- “固定 logits + 固定参数 + 固定 seed => 固定输出序列”

这样才真正贴近 llama 官方思路。

---

## 7. 一句话总结

当前仓库的实现是：

> **固定离线 token 表 + 线性同余随机 + 取模选择**

而 llama 官方在这块的思路是：

> **模型先产出 logits，sampling chain 对候选分布做约束和变换，最后再基于 seed 从概率分布中采样 token**

所以如果整个工程必须依赖 llama 官方思路，那么后续 Python 和 C 的实现方向都应该从“随机取表项”转向“基于 logits 的 sampler chain”。

---

## 8. 参考位置

### 本仓库

- `src/random_token.c`
- `tools/golden_verify.py`

### llama.cpp 官方实现参考

- `common/sampling.cpp`：公共 sampling 初始化与采样流程
- `common/common.h`：sampling 参数定义（如 `seed`、`top_k`、`top_p`、`min_p`、`temp`）

可重点关注的官方思路包括：

- sampler chain 初始化
- grammar / penalty / temperature / top-k / top-p / min-p 的串联
- 默认 `dist sampler` 作为最终抽样器
- 采样后对 sampler 状态的 accept / 更新
