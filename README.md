# RL by Doing

这是一套按实践推进的强化学习学习记录。当前只发布已经学完并整理好的内容。

## 已发布内容

### Lesson 01: Multi-Armed Bandit

第一课用多臂老虎机问题理解强化学习里的第一个核心矛盾：

```text
探索 vs 利用
```

你会学到：

- 为什么 Bandit 是 RL 的入门问题；
- 为什么只有历史数据还不一定适合直接用传统 ML；
- `greedy`、`epsilon_greedy`、`ucb` 三种策略的区别；
- 如何阅读实验输出里的 reward、regret、best action rate；
- 如何把代码变量对应到 `Q(a)`、`N(a)`、`r` 这些基本概念；
- 为什么 UCB 是“估计收益 + 不确定性奖励”的一类方法。

材料：

- [第一课学习入口](lessons/01_bandit/README.md)
- [第一课分享版教程](docs/articles/lesson_01_bandit_tutorial.md)
- [第一课代码](lessons/01_bandit/bandit.py)

## 运行第一课

当前第一课只使用 Python 标准库。

```bash
python lessons/01_bandit/bandit.py
```

也可以运行参数实验：

```bash
python lessons/01_bandit/bandit.py --epsilon 0.01
python lessons/01_bandit/bandit.py --epsilon 0.3
python lessons/01_bandit/bandit.py --steps 200
python lessons/01_bandit/bandit.py --steps 10000
python lessons/01_bandit/bandit.py --reward-std 2.0
python lessons/01_bandit/bandit.py --ucb-c 0.5
python lessons/01_bandit/bandit.py --ucb-c 4.0
```

## 发布节奏

后续会按“一课学完、一课整理、一课发布”的方式继续更新。
