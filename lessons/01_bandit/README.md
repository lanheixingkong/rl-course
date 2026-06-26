# Lesson 01: Multi-Armed Bandit

本课的目标：通过一个最小可运行案例，理解 RL 的第一个核心矛盾：探索和利用。

## 这一课解决什么问题

你要在多个选项里反复选择一个，例如广告、推荐内容、实验方案或药物剂量。每次选择后，只能看到被选动作的收益，看不到其他动作如果被选会怎样。

这就是 RL 里最小的“边试边学”问题。

它不是因为算法复杂才适合作为 RL 入门，而是因为它保留了 RL 的核心数据生成方式：agent 的动作会决定它接下来能看到什么反馈。传统监督学习通常假设数据已经给定，而 bandit 要同时处理“现在选什么”和“如何收集更有价值的数据”。

一个实用判断：如果已有数据完整、覆盖充分、偏差可控，优先考虑传统 ML；如果只能通过行动获得反馈，并且未选择动作没有标签，就进入 Bandit/RL 更擅长处理的范围。

## 运行

```bash
python lessons/01_bandit/bandit.py
```

你会看到类似这样的输出：

```text
True action values: [0.2, 0.0, 1.4, 0.7, 1.0]
Best action: 2

Averaged over 100 independent runs, 2000 steps each.
epsilon=0.1, reward_std=1.0, ucb_c=2.0

Strategy: greedy
  avg total reward     :  1991.53
  avg regret           :   808.47
  avg best action rate :   35.95%
  last-run estimates   : [-0.17, -0.02, 1.42, 0.0, 0.0]
  last-run counts      : [1, 21, 1978, 0, 0]

Strategy: epsilon_greedy
  avg total reward     :  2604.36
  avg regret           :   195.64
  avg best action rate :   86.82%
  last-run estimates   : [0.17, -0.14, 1.43, 0.61, 1.06]
  last-run counts      : [48, 23, 1752, 47, 130]

Strategy: ucb
  avg total reward     :  2683.85
  avg regret           :   116.15
  avg best action rate :   90.52%
  last-run estimates   : [0.25, 0.09, 1.43, 0.61, 0.77]
  last-run counts      : [19, 17, 1879, 35, 50]
```

### 先别急着理解所有细节

第一次运行后，你只需要先看懂一句话：

```text
这个程序在比较 3 种“选择动作的方法”，看哪一种长期拿到的奖励更多。
```

在这个例子里，有 5 个动作，你可以把它们理解成 5 个广告、5 个推荐策略、5 个实验方案，或者 5 台老虎机。每一步只能选其中一个动作，然后得到一个奖励。

### 输出第一部分是什么意思

```text
True action values: [0.2, 0.0, 1.4, 0.7, 1.0]
Best action: 2
```

这两行是“上帝视角”，真实业务里通常不知道。这里为了教学，程序故意告诉我们：

- 一共有 5 个动作，编号是 `0, 1, 2, 3, 4`。
- 每个动作背后有一个真实平均收益。
- 动作 `2` 的真实平均收益是 `1.4`，最高，所以它是真正最好的动作。

agent 一开始不知道这些真实值，只能靠不断尝试来估计。

```text
Averaged over 100 independent runs, 2000 steps each.
```

意思是：为了避免一次实验太偶然，程序重复做了 100 次实验；每次实验里，agent 连续选择 2000 次动作。最后输出的是 100 次实验的平均结果。

这里最容易混淆的是“5 个动作”和“2000 步”的关系：

```text
5 个动作 = 每一步可以选择的 5 个选项
2000 步 = 重复做 2000 次选择
```

不是有 2000 个不同动作，而是有 5 个固定动作，agent 要在这 5 个动作里反复选 2000 次。

可以把它想成 5 台老虎机：

```text
第 1 步：从 5 台老虎机里选 1 台拉一下
第 2 步：再从同样 5 台老虎机里选 1 台拉一下
第 3 步：再选 1 台
...
第 2000 步：再选 1 台
```

每一步都只能选一个动作。选完后，程序会给这个动作一个随机奖励。agent 用这些历史奖励慢慢判断哪一个动作更值得选。

例如，`last-run counts` 可能是：

```text
[48, 23, 1752, 47, 130]
```

这 5 个数字加起来正好是 2000：

```text
48 + 23 + 1752 + 47 + 130 = 2000
```

意思是：在最后一次实验的 2000 步里，动作 0 被选了 48 次，动作 1 被选了 23 次，动作 2 被选了 1752 次，动作 3 被选了 47 次，动作 4 被选了 130 次。

### 三个策略是什么意思

`Strategy` 表示“选择动作的策略”，也就是 agent 每一步如何决定选哪个动作。

| 名称 | 中文理解 | 它怎么选 |
| --- | --- | --- |
| `greedy` | 贪心策略 | 永远选当前看起来平均奖励最高的动作 |
| `epsilon_greedy` | 带一点随机探索的贪心策略 | 大多数时候选当前最好，少数时候随机试试别的动作 |
| `ucb` | 乐观探索策略 | 不只看当前平均奖励，也优先尝试那些试得少、还不确定的动作 |

这里的 `greedy` 不是贬义词，只是算法名。它的意思是“只利用当前已有信息，不主动探索”。

`epsilon_greedy` 里的 `epsilon` 表示随机探索的比例。默认 `epsilon=0.1`，意思是：

```text
90% 的时候选当前看起来最好的动作；
10% 的时候随机选一个动作，用来探索。
```

`ucb` 可以先理解为：如果一个动作还没怎么试过，就先不要过早判死刑，要给它一些机会。

### 每个指标是什么意思

每个策略下面都有 5 行指标：

```text
avg total reward
avg regret
avg best action rate
last-run estimates
last-run counts
```

逐个解释：

| 指标 | 中文含义 | 怎么看 |
| --- | --- | --- |
| `avg total reward` | 平均总奖励 | 越高越好，表示长期拿到的收益越多 |
| `avg regret` | 平均后悔值 / 损失 | 越低越好，表示离理论最优越近 |
| `avg best action rate` | 选中真实最优动作的比例 | 越高越好 |
| `last-run estimates` | 最后一次实验中，agent 对每个动作平均收益的估计 | 用来看 agent 学到了什么 |
| `last-run counts` | 最后一次实验中，每个动作被选了多少次 | 用来看 agent 把机会给了哪些动作 |

`regret` 可以先理解成：

```text
如果我每次都选真正最好的动作，本来能拿多少奖励；
现在因为我还不够了解环境，少拿了多少。
```

所以 `regret` 不是情绪上的后悔，而是一个数字指标。

### 怎么读默认结果

默认结果里最重要的是这三行：

```text
greedy           avg total reward = 1991.53, avg regret = 808.47
epsilon_greedy   avg total reward = 2604.36, avg regret = 195.64
ucb              avg total reward = 2683.85, avg regret = 116.15
```

你应该得到三个结论：

1. `greedy` 最差，因为它太早相信“当前看起来最好”的动作。
2. `epsilon_greedy` 更好，因为它保留了一点随机探索，能发现更好的动作。
3. `ucb` 在这个实验里最好，因为它更聪明地探索那些“不确定但可能好”的动作。

再看 `greedy` 的最后一次实验：

```text
last-run counts: [1, 21, 1978, 0, 0]
```

这表示在最后一次实验的 2000 步里：

- 动作 0 被选了 1 次；
- 动作 1 被选了 21 次；
- 动作 2 被选了 1978 次；
- 动作 3 和 4 一次都没选。

这次 `greedy` 运气不错，后来几乎一直选中了最优动作 2。但它的平均结果仍然很差，说明很多其他实验里，`greedy` 会因为早期随机奖励误判，然后长期卡在错误动作上。

这就是本课第一核心：**只利用当前看起来最好的选择，可能会被早期噪声误导；适度探索能减少长期损失。**

## 改实验参数：每次只验证一个问题

改参数不是为了追求某个固定答案，而是为了验证 RL 里的几个核心现象。每次只改一个参数，其他参数保持不变，这样你才能知道输出变化是由哪个因素造成的。

### 实验 1：改 `epsilon`，验证“探索太少或太多都不好”

`epsilon` 只影响 `epsilon_greedy` 策略。

它表示“随机探索的比例”：

```text
epsilon = 0.1
```

意思是：

```text
90% 的时候选当前看起来最好的动作；
10% 的时候随机选一个动作。
```

运行：

```bash
python lessons/01_bandit/bandit.py --epsilon 0.01
python lessons/01_bandit/bandit.py --epsilon 0.3
```

这两个命令是在问两个问题：

```text
epsilon = 0.01: 如果只用 1% 的机会探索，会不会太保守？
epsilon = 0.3 : 如果用 30% 的机会随机探索，会不会浪费太多机会？
```

你重点看 `epsilon_greedy` 这一组输出里的三行：

```text
avg total reward
avg regret
avg best action rate
```

预期现象：

- `epsilon=0.01` 通常探索太少，容易早早卡在错误动作上。
- `epsilon=0.3` 通常探索太多，即使知道哪个动作好，也经常随机选到差动作。
- 默认 `epsilon=0.1` 在这个例子里比较平衡。

注意：`greedy` 和 `ucb` 不使用 `epsilon`，所以你改 `epsilon` 时，它们的结果不会因为这个参数而变化。

### 实验 2：改 `steps`，验证“任务越长，早期探索越值得”

`steps` 表示一轮实验里，agent 要连续做多少次选择。

```text
steps = 2000
```

意思是：在同样 5 个动作里，重复选择 2000 次。

运行：

```bash
python lessons/01_bandit/bandit.py --steps 200
python lessons/01_bandit/bandit.py --steps 10000
```

这两个命令是在问：

```text
steps = 200  : 如果任务很短，探索还有没有足够时间回本？
steps = 10000: 如果任务很长，前期探索找到好动作后，后面能不能长期受益？
```

你重点比较三种策略的：

```text
avg regret
avg best action rate
```

预期现象：

- 步数很短时，探索的好处还没完全体现出来。
- 步数很长时，能尽早发现好动作的策略更占优势。
- `greedy` 如果早期选错，错误会被放大到很长时间。

### 实验 3：改 `reward-std`，验证“反馈越吵，越不能相信早期结果”

`reward-std` 表示奖励的随机波动大小。`std` 是 standard deviation，也就是标准差。

默认：

```text
reward_std = 1.0
```

如果改成：

```text
reward_std = 2.0
```

意思是：每次奖励更不稳定，同一个动作有时奖励高、有时奖励低。

运行：

```bash
python lessons/01_bandit/bandit.py --reward-std 2.0
```

这个命令是在问：

```text
如果反馈更随机，agent 是否更容易被少量早期样本误导？
```

你重点看：

```text
avg regret
last-run estimates
last-run counts
```

预期现象：

- 奖励噪声越大，单次反馈越不可靠。
- agent 需要更多尝试，才能判断一个动作是真的好，还是只是这次运气好。
- 不探索或探索不足的策略更容易被噪声影响。

### 实验 4：改 `ucb-c`，验证“UCB 的探索强度也需要控制”

`ucb-c` 只影响 `ucb` 策略。

在 UCB 里，每个动作的分数大致是：

```text
分数 = 当前估计收益 + 探索奖励
```

代码里对应：

```text
estimate + c * sqrt(log(step + 1) / count)
```

这里的 `c` 就是命令行参数 `--ucb-c`。它控制探索奖励有多强。

默认：

```text
ucb_c = 2.0
```

运行：

```bash
python lessons/01_bandit/bandit.py --ucb-c 0.5
python lessons/01_bandit/bandit.py --ucb-c 4.0
```

这两个命令是在问：

```text
ucb-c = 0.5: 如果 UCB 探索奖励变小，会不会更快利用当前好动作？
ucb-c = 4.0: 如果 UCB 探索奖励变大，会不会探索过多？
```

你重点看 `ucb` 这一组输出里的：

```text
avg total reward
avg regret
avg best action rate
last-run counts
```

在当前固定案例里，一组参考结果是：

| ucb-c | UCB 平均总收益 | UCB 平均 regret | UCB 最优动作选择率 |
| ---: | ---: | ---: | ---: |
| 0.5 | 2704.82 | 95.18 | 89.59% |
| 2.0 | 2683.85 | 116.15 | 90.52% |
| 4.0 | 2478.10 | 321.90 | 75.44% |

这个结果说明：UCB 也不是探索越多越好。`ucb-c` 太大时，会给“不确定动作”过高的探索奖励，导致 agent 明明已经知道动作 2 很好，却还频繁尝试较差动作。

但也要注意：`ucb-c=0.5` 在这个小例子里表现更好，不代表它永远更好。不同环境、奖励噪声、动作差距下，合适的探索强度可能不同。

## 观察重点

输出里会比较三种策略在多次实验中的平均表现：

- `greedy`: 贪心策略，永远选择当前平均收益最高的动作。
- `epsilon_greedy`: 带一点随机探索的贪心策略，大多数时候选当前最好，少数时候随机探索。
- `ucb`: 乐观探索策略，优先选择“不确定但可能好”的动作。

重点看：

- 累计奖励谁更高；
- 是否找到了真实最优臂；
- regret 是否持续增长。
- greedy 为什么有时看起来很好，但平均下来不稳定。

默认实验的关键结果：

| 策略 | 平均总收益 | 平均 regret | 最优动作选择率 |
| --- | ---: | ---: | ---: |
| greedy | 1991.53 | 808.47 | 35.95% |
| epsilon-greedy, epsilon=0.1 | 2604.36 | 195.64 | 86.82% |
| UCB | 2683.85 | 116.15 | 90.52% |

这个结果说明：只利用当前看起来最好的动作，很容易被早期随机奖励误导；适度探索能显著提升长期收益。

## 代码和原理对照

公式不要单独背。先看代码里的变量，再把它翻译成 RL 术语。

### 1. 环境里真实存在什么

在 `main()` 里：

```python
true_means = [0.2, 0.0, 1.4, 0.7, 1.0]
```

这表示 5 个动作的真实平均奖励。

| 动作编号 | 真实平均奖励 |
| ---: | ---: |
| 0 | 0.2 |
| 1 | 0.0 |
| 2 | 1.4 |
| 3 | 0.7 |
| 4 | 1.0 |

动作 2 的真实平均奖励最高，所以真实最优动作是 2。

但注意：这是程序里的“上帝视角”。agent 一开始不知道 `true_means`，它只能通过试错估计哪个动作更好。

在 RL 术语里，动作 `a` 的真实平均奖励通常写成：

```text
q*(a)
```

你可以先把它读成：

```text
动作 a 真正平均能带来多少奖励
```

### 2. agent 自己相信什么

在 `run_bandit()` 里：

```python
estimates = [0.0 for _ in range(bandit.n_actions)]
counts = [0 for _ in range(bandit.n_actions)]
```

`estimates` 是 agent 对每个动作平均奖励的估计。

刚开始：

```text
estimates = [0, 0, 0, 0, 0]
```

意思是：agent 对 5 个动作还一无所知，所以先都估计成 0。

`counts` 是每个动作被选过多少次。

刚开始：

```text
counts = [0, 0, 0, 0, 0]
```

意思是：5 个动作都还没试过。

在 RL 术语里，agent 对动作 `a` 的估计值通常写成：

```text
Q(a)
```

你可以先把它读成：

```text
我目前估计动作 a 平均能带来多少奖励
```

所以：

```text
q*(a) = 真实答案，agent 不知道
Q(a)  = agent 当前估计，agent 会不断更新
```

### 3. agent 如何选择动作

每一步都会走到这段代码：

```python
if name == "greedy":
    action = choose_greedy(estimates, rng)
elif name == "epsilon_greedy":
    action = choose_epsilon_greedy(estimates, epsilon, rng)
elif name == "ucb":
    action = choose_ucb(estimates, counts, step, ucb_c)
```

这段代码的意思是：根据当前策略，选择一个动作。

三种策略的区别是：

- `greedy`: 只看 `estimates`，选当前估计值最大的动作。
- `epsilon_greedy`: 大多数时候像 `greedy`，少数时候随机探索。
- `ucb`: 同时看 `estimates` 和 `counts`，优先给“试得少、还不确定”的动作一些机会。

这就是 RL 里的 policy，也就是策略：

```text
policy = 根据当前信息决定选哪个动作的方法
```

### 4. `greedy` 策略怎么实现

代码：

```python
def choose_greedy(estimates: list[float], rng: random.Random) -> int:
    best_value = max(estimates)
    candidates = [i for i, value in enumerate(estimates) if value == best_value]
    return rng.choice(candidates)
```

假设当前估计是：

```text
estimates = [0.2, 0.0, 1.4, 0.7, 1.0]
```

第一行：

```python
best_value = max(estimates)
```

找到当前估计里最大的值，也就是 `1.4`。

第二行：

```python
candidates = [i for i, value in enumerate(estimates) if value == best_value]
```

找到所有估计值等于最大值的动作编号。这里得到：

```text
candidates = [2]
```

如果多个动作并列第一，例如：

```text
estimates = [0.0, 0.0, 0.0, 0.0, 0.0]
```

那 `candidates` 就是：

```text
[0, 1, 2, 3, 4]
```

第三行：

```python
return rng.choice(candidates)
```

从候选动作里随机选一个。这样做是为了处理“并列第一”的情况。

`greedy` 的核心特点：

```text
只看当前估计值 estimates
不主动尝试没试过或试得少的动作
```

所以它可能因为早期几次随机奖励而过早相信某个动作。

### 5. `epsilon_greedy` 策略怎么实现

代码：

```python
def choose_epsilon_greedy(estimates: list[float], epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(estimates))
    return choose_greedy(estimates, rng)
```

第一行判断：

```python
if rng.random() < epsilon:
```

`rng.random()` 会生成一个 0 到 1 之间的随机数。

如果 `epsilon=0.1`，那么大约 10% 的时候，这个随机数会小于 0.1，于是进入探索分支：

```python
return rng.randrange(len(estimates))
```

`len(estimates)` 是动作数量，这里是 5。所以这行会在动作 `0, 1, 2, 3, 4` 里随机选一个。

如果没有进入探索分支，就执行：

```python
return choose_greedy(estimates, rng)
```

也就是按 greedy 方式选当前估计最好的动作。

所以 `epsilon_greedy` 可以翻译成：

```text
少数时候随机探索；
大多数时候选择当前看起来最好的动作。
```

它比 `greedy` 多了一点探索能力，但探索是随机的，不会区分“哪个动作更值得探索”。

### 6. `ucb` 策略怎么实现

代码：

```python
def choose_ucb(estimates: list[float], counts: list[int], step: int, c: float) -> int:
    for action, count in enumerate(counts):
        if count == 0:
            return action

    scores = [
        estimate + c * math.sqrt(math.log(step + 1) / count)
        for estimate, count in zip(estimates, counts)
    ]
    return scores.index(max(scores))
```

先看第一段：

```python
for action, count in enumerate(counts):
    if count == 0:
        return action
```

这表示：如果某个动作一次都没试过，就先试它。

原因很直接：一个动作如果从来没试过，agent 根本不知道它好不好。UCB 不会让某个动作永远没有机会。

等所有动作至少都试过一次后，进入第二段：

```python
scores = [
    estimate + c * math.sqrt(math.log(step + 1) / count)
    for estimate, count in zip(estimates, counts)
]
```

这段会给每个动作计算一个分数：

```text
score = 当前估计值 + 探索奖励
```

对应代码：

```text
estimate + c * sqrt(log(step + 1) / count)
```

它由两部分组成：

| 代码部分 | 直觉含义 |
| --- | --- |
| `estimate` | 这个动作当前看起来有多好 |
| `count` | 这个动作已经试过多少次 |
| `sqrt(log(step + 1) / count)` | 试得越少，不确定性越大，探索奖励越高 |
| `c` | 控制探索奖励的强弱 |

#### 为什么是“估计值 + 探索奖励”

UCB 的全称可以理解成 Upper Confidence Bound，也就是“置信上界”。

它的想法不是只问：

```text
这个动作当前平均收益是多少？
```

而是问：

```text
在乐观估计下，这个动作有没有可能其实很好？
```

所以它给每个动作算一个偏乐观的分数：

```text
UCB 分数 = 当前估计收益 + 不确定性奖励
```

如果一个动作当前估计收益高，它会被选中。  
如果一个动作试得很少、还不确定，它也会被加分，因为它“可能被低估了”。

#### 为什么要除以 `count`

公式里的这一部分：

```text
1 / count
```

表达的是：

```text
一个动作试得越多，我们对它越有把握；
一个动作试得越少，我们越不确定。
```

所以当 `count` 很小时，探索奖励大；当 `count` 越来越大时，探索奖励会变小。

这符合直觉：动作 2 已经试了 1000 次，就没必要再因为“不确定”给它很多额外奖励；动作 4 只试了 3 次，就还值得再看看。

#### 为什么有 `log(step + 1)`

公式里的：

```text
log(step + 1)
```

表示：随着总决策次数增加，系统会慢慢提高对探索的要求。

直觉是：如果已经进行了很多步，而某个动作仍然试得很少，那么它的不确定性就更值得被注意。

但这里用的是 `log`，不是直接用 `step`。原因是探索压力应该慢慢增长，而不是增长得太快。`log` 增长很慢，所以它只是温和地提醒 agent：

```text
时间过去越久，别完全忘了那些试得少的动作。
```

#### 为什么外面还有 `sqrt`

`sqrt(...)` 会让探索奖励下降得更平滑。

你不用记数学证明，只要先记住直觉：

```text
count 越大，不确定性越小；
但不确定性不是线性下降，而是逐渐变小。
```

这个形式来自经典 UCB1 算法背后的概率界。后面如果深入 bandit 理论，会看到它和 Hoeffding bound 这类集中不等式有关。当前阶段不需要推导，只需要理解它在代码里起什么作用。

#### 所有 UCB 都用这个公式吗

不是。

UCB 是一类方法，不是唯一一个固定公式。共同思想是：

```text
选择“估计收益 + 不确定性奖励”最高的动作
```

但“不确定性奖励”可以有不同写法。常见变体包括：

- UCB1：常见形式是 `Q(a) + c * sqrt(log(t) / N(a))`。
- Gaussian UCB：如果假设奖励噪声近似高斯，可以用和方差相关的上界。
- Bayesian UCB：用后验分布的高分位数作为乐观估计。
- Linear UCB：有上下文特征时，用线性模型估计奖励和不确定性。

本课代码使用的是最适合入门的 UCB1 风格：

```text
estimate + c * sqrt(log(step + 1) / count)
```

它足够简单，同时能体现 UCB 的核心思想：**不是随机探索，而是按不确定性探索。**

最后：

```python
return scores.index(max(scores))
```

选择分数最高的动作。

UCB 的核心特点：

```text
不只看“现在估计收益高不高”，
还看“这个动作是不是试得太少、还不确定”。
```

所以它比 `epsilon_greedy` 更有方向感：不是完全随机探索，而是优先探索那些“不确定但可能值得”的动作。

### 7. 三种策略放在一起比较

| 策略 | 看哪些信息 | 怎么探索 | 主要风险 |
| --- | --- | --- | --- |
| `greedy` | 只看 `estimates` | 不主动探索 | 早期误判后长期卡住 |
| `epsilon_greedy` | 主要看 `estimates` | 按固定概率随机探索 | 探索可能太少或太多 |
| `ucb` | 同时看 `estimates` 和 `counts` | 优先探索试得少的动作 | 公式稍复杂，需要调 `c` |

这一节的重点不是背 UCB 公式，而是理解三种策略的差别：

```text
greedy: 只利用
epsilon_greedy: 随机探索 + 利用
ucb: 根据不确定性探索 + 利用
```

### 8. 选完动作后，环境给奖励

代码：

```python
reward = bandit.pull(action)
```

意思是：agent 选择了 `action`，环境返回一个奖励 `reward`。

这个奖励不是固定值，而是围绕真实平均奖励上下波动。比如动作 2 的真实平均奖励是 1.4，但某一次可能拿到 0.8，另一次可能拿到 2.1。

这就是为什么 agent 不能只看一次结果就下结论。

### 9. agent 更新这个动作的估计

核心代码是这两行：

```python
counts[action] += 1
estimates[action] += (reward - estimates[action]) / counts[action]
```

第一行：

```python
counts[action] += 1
```

意思是：这个动作又被试了一次。

第二行：

```python
estimates[action] += (reward - estimates[action]) / counts[action]
```

意思是：用这次新拿到的奖励，修正对这个动作的平均奖励估计。

可以拆开看：

```text
reward - estimates[action]
```

表示“这次奖励”和“旧估计”之间差多少。

```text
(reward - estimates[action]) / counts[action]
```

表示这次要把旧估计往新奖励方向调整多少。这个动作试得越多，`counts[action]` 越大，单次新奖励对估计的影响越小。

对应到公式就是：

```text
Q(a) <- Q(a) + (r - Q(a)) / N(a)
```

先不用害怕这个公式。它只是代码这一行的数学写法：

```python
estimates[action] += (reward - estimates[action]) / counts[action]
```

变量对照：

| 公式 | 代码 | 含义 |
| --- | --- | --- |
| `a` | `action` | 这一步选择的动作 |
| `r` | `reward` | 这一步拿到的奖励 |
| `Q(a)` | `estimates[action]` | agent 对这个动作平均奖励的估计 |
| `N(a)` | `counts[action]` | 这个动作被选过多少次 |

#### 这个更新公式是固定的吗

不是所有 RL 都固定使用这个公式。

这个公式是“样本平均”的增量写法。它适合当前 bandit 例子，因为我们想估计：

```text
某个动作到目前为止拿到的平均奖励
```

如果一个动作的历史奖励是：

```text
1.0, 2.0, 3.0
```

平均值是：

```text
(1.0 + 2.0 + 3.0) / 3 = 2.0
```

但每次都重新保存所有历史奖励再求平均比较麻烦，所以代码用增量方式更新平均值：

```text
新平均值 = 旧平均值 + (新奖励 - 旧平均值) / 次数
```

也就是：

```text
Q(a) <- Q(a) + (r - Q(a)) / N(a)
```

更通用的写法是：

```text
新估计 = 旧估计 + 步长 * 误差
```

在本课里：

```text
误差 = r - Q(a)
步长 = 1 / N(a)
```

所以本课公式只是通用更新思想的一种具体形式。

后面学 Q-learning 时，你会看到很像的结构：

```text
Q(s,a) <- Q(s,a) + alpha * TD error
```

这里的 `alpha` 是学习率，不一定等于 `1 / N(a)`；而 `TD error` 也不只是 `r - Q(a)`，它还会包含下一状态的长期价值。

所以要记住的是这条主线：

```text
用新反馈产生一个误差，再用这个误差修正旧估计。
```

### 10. agent 如何统计总收益

代码：

```python
total_reward += reward
best_action_count += int(action == bandit.best_action)
```

`total_reward` 是这一轮实验的总奖励。它回答：

```text
这 2000 次选择一共赚了多少？
```

`best_action_count` 是选中真实最优动作的次数。它回答：

```text
这 2000 次选择里，有多少次选中了动作 2？
```

输出里的 `avg best action rate` 就来自这里。

### 11. regret 是怎么从代码算出来的

regret 表示“因为没有总选真实最优动作而损失了多少收益”：

```text
regret = 理论最优总收益 - 实际总收益
```

代码：

```python
optimal_reward = steps * bandit.best_mean
regret = optimal_reward - total_reward
```

这里：

```text
steps = 2000
bandit.best_mean = 1.4
```

所以理论最优总收益是：

```text
2000 * 1.4 = 2800
```

如果某个策略实际拿到 2604.36，那么 regret 大约是：

```text
2800 - 2604.36 = 195.64
```

regret 越低，说明这个策略离“每次都选真实最优动作”的理想情况越近。

### 12. 本课最重要的代码线索

读代码时，抓住这条主线：

```text
选择动作 -> 拿到奖励 -> 更新估计 -> 下一步用新估计继续选择
```

对应代码：

```python
action = choose_...(estimates, ...)
reward = bandit.pull(action)
counts[action] += 1
estimates[action] += (reward - estimates[action]) / counts[action]
```

这就是“边尝试边学习”的最小版本。

## 练习

1. 把 `epsilon` 从 `0.1` 改成 `0.01` 和 `0.3`，观察长期收益。
2. 把 `steps` 改成 `200` 和 `10000`，看短期和长期哪个策略更好。
3. 修改 `reward_std`，让奖励噪声变大，观察探索策略是否更重要。

## 分享版教程

完整教程见：[docs/articles/lesson_01_bandit_tutorial.md](../../docs/articles/lesson_01_bandit_tutorial.md)。

其中第 11-14 节专门回答：Bandit 为什么是 RL 入门问题、能不能用传统机器学习或神经网络解决、什么时候不该用 RL。
