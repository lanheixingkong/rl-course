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

### Lesson 02: GridWorld Dynamic Programming

第二课用 GridWorld 理解完整 RL 问题里的几个核心对象：

```text
state -> action -> reward -> value -> policy
```

你会学到：

- GridWorld 相比 Bandit 多了什么；
- `Values` 表和 `Policy` 表分别表示什么；
- 为什么本课属于 planning：环境模型已知，可以直接推算；
- policy evaluation 如何评估当前策略的长期价值；
- policy improvement 如何根据价值改进策略；
- 为什么 `V(s) = r + gamma * V(s')` 是价值定义的递归展开；
- 为什么环境未知时需要进入 Q-learning 这类从经验学习的方法。

材料：

- [第二课学习入口](lessons/02_gridworld_dp/README.md)
- [第二课分享版教程](docs/articles/lesson_02_gridworld_dp_tutorial.md)
- [第二课封面图](docs/articles/lesson_02_gridworld_dp_cover.png)
- [第二课代码](lessons/02_gridworld_dp/gridworld_dp.py)

### Lesson 03: Q-learning GridWorld

第三课继续使用 GridWorld，但去掉“提前用完整环境模型做 planning”的学习方式，改为通过一条条经验更新动作价值：

```text
state, action, reward, next_state -> Q(s,a)
```

你会学到：

- 第二课的 `V(s)` 和第三课的 `Q(s,a)` 有什么区别；
- Q-learning 如何用单条经验更新动作价值；
- `target`、`td error`、`alpha` 分别表示什么；
- 为什么 `target` 早期可能看起来等于 `step_reward`；
- 为什么训练时会探索，但学习目标仍然朝 greedy 策略靠近；
- 为什么 Q-learning 被称为 model-free 和 off-policy。

材料：

- [第三课学习入口](lessons/03_q_learning/README.md)
- [第三课分享版教程](docs/articles/lesson_03_q_learning_tutorial.md)
- [第三课封面图](docs/articles/lesson_03_q_learning_cover.png)
- [第三课代码](lessons/03_q_learning/q_learning_gridworld.py)

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

## 运行第二课

当前第二课只使用 Python 标准库。

```bash
python lessons/02_gridworld_dp/gridworld_dp.py
```

如果想观察 `evaluate_policy()` 的价值更新过程：

```bash
python lessons/02_gridworld_dp/gridworld_dp.py --show-evaluation --debug-sweeps 2
```

也可以运行参数实验：

```bash
python lessons/02_gridworld_dp/gridworld_dp.py --gamma 0.5
python lessons/02_gridworld_dp/gridworld_dp.py --step-reward -0.2
python lessons/02_gridworld_dp/gridworld_dp.py --step-reward -1.0
python lessons/02_gridworld_dp/gridworld_dp.py --pit-reward -5.0
```

## 运行第三课

当前第三课只使用 Python 标准库。

```bash
python lessons/03_q_learning/q_learning_gridworld.py
```

如果想观察单步 Q-learning 更新：

```bash
python lessons/03_q_learning/q_learning_gridworld.py --episodes 5 --debug-episodes 1 --log-every 0
```

也可以运行参数实验：

```bash
python lessons/03_q_learning/q_learning_gridworld.py --episodes 100 --log-every 50
python lessons/03_q_learning/q_learning_gridworld.py --episodes 500 --epsilon 0 --log-every 250
python lessons/03_q_learning/q_learning_gridworld.py --episodes 500 --epsilon 0.5 --log-every 250
python lessons/03_q_learning/q_learning_gridworld.py --episodes 1000 --alpha 0.05 --log-every 500
python lessons/03_q_learning/q_learning_gridworld.py --slip-probability 0.1 --log-every 1000
```

## 发布节奏

后续会按“一课学完、一课整理、一课发布”的方式继续更新。
