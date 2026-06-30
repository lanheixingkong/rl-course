# 第 0 课：这套强化学习课程的总览地图

学完前三课后，单独看每一课都能理解，但如果不知道它们为什么这样排列，就容易产生一个问题：

> 我现在学到的这些东西，在强化学习整体里到底处在什么位置？

这篇第 0 课不是正式算法课，也没有代码。它的作用是给这套课程一张粗略地图：先知道我们从哪里开始，为什么这样走，后面大概会走向哪里。

这张地图不是最终版。后面的课程会根据实际学习过程调整，但它能帮助你在当前阶段建立一个整体参照。

## 这套课不是按历史顺序，而是按概念依赖顺序

强化学习的发展历史很长，里面有动态规划、蒙特卡洛方法、时序差分学习、Q-learning、策略梯度、Actor-Critic、DQN、PPO、AlphaGo/AlphaZero 等很多路线。

如果按历史顺序学，很容易变成“先学一堆公式和年代，再慢慢等到实际问题”。这和本课程的目标不一致。

这套课采用的是另一种顺序：

```text
先学最小可运行决策问题
再逐步增加一个新的复杂度
每增加一个复杂度，就引入一个必要的新概念
```

前三课的顺序可以理解成：

```text
Bandit
  -> 加入 state，变成完整 MDP 问题
  -> 如果环境模型已知，可以 planning：Dynamic Programming
  -> 如果环境模型未知，只能从经验学习：Q-learning
```

所以，前三课不是在复刻强化学习历史，而是在搭建后面所有课程都会反复用到的概念骨架。

## 强化学习最核心的一句话

强化学习研究的问题可以先粗略理解为：

```text
Agent 在某个 state 下选择 action，
Environment 返回 next_state 和 reward，
Agent 根据长期累计 reward 的好坏，
逐步改进自己的 policy。
```

这里有几个关键词：

- `state`：当前情况是什么；
- `action`：现在可以做什么；
- `reward`：这一步做完以后环境给的数字反馈；
- `next_state`：动作之后进入什么新情况；
- `policy`：在不同 state 下应该选择什么 action；
- `value`：某个 state 或 action 长期看有多好。

后面几乎所有 RL 方法，都是围绕这些词在变化。

## 为什么第一课是 Bandit

Bandit 是最小的强化学习式决策问题。

它暂时没有复杂的 `state`，只有一组选项：

```text
每一步从多个 action 中选一个
选完以后得到 reward
不知道每个 action 的真实平均收益
需要一边尝试，一边决定以后选什么
```

它让你先看见 RL 里最早必须面对的矛盾：

```text
探索 exploration：试试看不熟悉的 action，可能发现更好的选择
利用 exploitation：选择当前看起来最好的 action，尽快获得收益
```

这就是为什么第一课不直接上深度神经网络。因为如果还没理解“为什么要探索”，后面看到 DQN、PPO、AlphaZero 里的探索策略时，就只能记术语，很难判断方法是否合理。

第一课学到的是：

```text
action value：Q(a)
exploration vs exploitation
greedy / epsilon-greedy / UCB
partial feedback：每次只知道被选 action 的结果
```

它回答的问题是：

> 当我只能边尝试边收集反馈时，怎样在“继续试”和“选当前最好”之间做权衡？

## 为什么第二课是 GridWorld DP

Bandit 太简单，因为它没有状态变化。

真实问题通常不是“永远在同一个地方选按钮”，而是：

```text
我现在处在一个状态
做一个动作
环境进入下一个状态
未来还能继续做动作
当前选择会影响后续处境
```

这就是第二课 GridWorld 引入的东西。

在 GridWorld 里，Agent 在格子地图中移动。每个格子是一个 `state`，上下左右是 `action`，移动后会得到 `reward`，最后形成一条路径。

第二课有一个重要前提：

```text
环境模型已知
```

也就是说，程序知道：

```text
在 state s 执行 action a，
会到达哪个 next_state，
会得到多少 reward。
```

因此它不需要真的一次次“走迷宫试错”，而是可以直接用规则推算当前策略的长期价值，再根据价值改进策略。这类问题叫 `planning`。

第二课学到的是：

```text
state
policy
V(s)
policy evaluation
policy improvement
Bellman-style self-consistency
planning with known model
```

它回答的问题是：

> 如果我知道环境规则，怎样通过推算长期价值来找更好的策略？

## 为什么第三课是 Q-learning

第二课虽然重要，但它有一个很强的假设：环境模型已知。

现实中很多问题没有这么方便。

比如推荐系统、机器人控制、游戏智能体、广告投放，Agent 往往并不能提前知道：

```text
如果我在这个状态做这个动作，下一步一定会发生什么？
```

它只能实际做一次动作，拿到一条经验：

```text
state, action, reward, next_state
```

第三课 Q-learning 就是在这个条件下学习。

它和第二课仍然使用 GridWorld，但学习方式变了：

```text
第二课：知道完整规则，遍历所有 state/action 做规划
第三课：只根据实际经历过的一条条经验更新 Q(s,a)
```

这就是 `model-free` 的基本含义：算法不依赖一个可以提前查询完整结果的环境模型，而是从交互经验中直接学习价值。

第三课学到的是：

```text
Q(s,a)
experience: (s, a, r, s')
TD target
TD error
alpha learning rate
model-free learning
off-policy learning
```

它回答的问题是：

> 如果我不知道完整环境规则，只能从经验中学习，怎样估计每个 state-action 的长期价值？

## 前三课共同构成什么

前三课不是三个孤立例子，而是在逐步补齐一套 RL 语言。

| 课程 | 新增的核心问题 | 学到的核心对象 |
| --- | --- | --- |
| Lesson 01 Bandit | 不知道哪个 action 好，必须探索 | `Q(a)`、探索/利用 |
| Lesson 02 GridWorld DP | action 会影响后续 state，且环境模型已知 | `V(s)`、`policy`、planning |
| Lesson 03 Q-learning | 环境模型未知，只能从经验学习 | `Q(s,a)`、TD 更新、model-free |

更压缩地看：

```text
Lesson 01：没有 state，只学习 action 好不好
Lesson 02：有 state，且知道环境规则，用 planning 推算
Lesson 03：有 state，但不知道完整规则，从经验更新 Q
```

这就是后面继续学习的地基。

## 用五个维度定位任何 RL 方法

以后看到一个新的 RL 方法时，不要先问“这个公式是什么意思”。先用下面五个问题定位它。

### 1. 它有没有 state？

如果没有复杂状态，只是在一组动作里反复选择，可能是 Bandit 问题。

如果当前动作会影响未来状态，那就是更完整的序列决策问题，通常会建模成 MDP。

### 2. 它是否知道环境模型？

环境模型指的是：

```text
给定 state 和 action，能否提前知道 next_state 和 reward？
```

如果知道，可以 planning，例如第二课的动态规划、游戏里的规则搜索、路径规划。

如果不知道，通常要从交互经验学习，例如 Q-learning、DQN、PPO。

还有一种中间路线：先学习一个环境模型，再用这个模型做 planning，这叫 model-based RL。

### 3. 它学习的是 value、policy，还是 model？

常见选择有三类：

```text
学习 value：判断某个 state 或 action 长期有多好
学习 policy：直接输出应该采取什么 action
学习 model：学习环境如何变化，再用模型规划
```

前三课主要在学 value：

```text
Lesson 01：Q(a)
Lesson 02：V(s)
Lesson 03：Q(s,a)
```

后面会学到直接学习 policy 的方法，以及同时学习 value 和 policy 的方法。

### 4. 它用表格还是神经网络表示？

前三课都是小问题，所以可以用表格：

```text
每个 state/action 都有一个明确的值
```

但真实问题的状态可能非常大：

```text
图像像素
连续传感器数据
用户画像
复杂游戏局面
机器人关节角度
```

这时不可能给每个状态都存一格表，就需要用函数逼近，最常见的是神经网络。

后面的 DQN 可以粗略理解为：

```text
Q-learning + neural network
```

### 5. 数据从哪里来？

RL 方法还要看数据来源：

```text
online interaction：Agent 一边行动一边收集数据
simulator：在仿真环境里大量试错
self-play：自己和自己对抗或练习
offline logs：只使用历史交互数据
```

同一个算法思想，在不同数据来源下，风险和实现方式会很不一样。

## 后续课程大概会学什么

当前路线会继续沿着“每次增加一个复杂度”的方式推进。

### Lesson 04：Monte Carlo 或 SARSA

这一步可能会补一个从完整 episode 回报中学习的方法，或者讲 SARSA。

它的作用是帮助你比较：

```text
Q-learning 为什么是 off-policy
SARSA 为什么是 on-policy
TD 学习和完整回报学习有什么区别
```

具体先讲 Monte Carlo 还是 SARSA，可以根据后面的学习情况调整。

### Lesson 05：Function Approximation

当状态空间变大后，表格方法不够用了。

这一课会讲：

```text
为什么不能再为每个 state/action 存一行表
怎样用模型近似 value 或 policy
传统 ML/DL 背景如何接入 RL
```

这一步是从表格 RL 走向深度 RL 的桥。

### Lesson 06：DQN

DQN 是一个很重要的深度 RL 入口。

它可以粗略理解为：

```text
Q-learning 的目标
+ 神经网络近似 Q(s,a)
+ experience replay
+ target network
```

你会看到为什么“直接把 Q-learning 换成神经网络”不稳定，以及 DQN 用了哪些工程机制让训练更可控。

### Lesson 07：Policy Gradient

Q-learning 学的是动作价值，然后通过价值选择动作。

Policy Gradient 换一种思路：

```text
直接学习 policy
直接让好动作的概率变大
让坏动作的概率变小
```

这对连续动作、随机策略、复杂控制问题很重要。

### Lesson 08：Actor-Critic / PPO

Actor-Critic 同时学习两个东西：

```text
Actor：负责选择 action，也就是 policy
Critic：负责评价 action 或 state，也就是 value
```

PPO 是现代实践中非常常见的一类策略优化方法。它不是第一个要学的算法，但学完 Q-learning、Policy Gradient、Actor-Critic 后，再看 PPO 会自然很多。

### 后续项目课：把 RL 用到实际问题

算法课之外，还需要练习如何建模真实问题：

```text
库存补货
广告/推荐策略
任务调度
机器人或控制问题
游戏 AI
资源分配
```

这类课程的重点不是背算法，而是回答：

```text
state 怎么定义？
action 怎么定义？
reward 怎么设计？
episode 什么时候结束？
数据从哪里来？
是否能安全探索？
是否应该用 RL，还是传统 ML 更合适？
```

## AlphaGo 和 AlphaZero 在地图上的位置

AlphaGo / AlphaZero 不是简单的第二课，也不是简单的第三课。

它们更像是多个方向的组合：

```text
已知游戏规则 -> 可以用规则做搜索和 planning
self-play -> 通过自己和自己下棋产生训练数据
neural network -> 学习 policy/value
MCTS -> 在当前局面下做搜索规划
```

所以它们和第二课相似的地方是：游戏规则已知，可以模拟未来变化。

它们和第三课相似的地方是：价值和策略不是手写出来的，而是从经验中学习出来的。

但它们又不同于前三课：它们不维护一个小表格 `Q(s,a)`，而是用神经网络表示复杂局面的策略和价值，并结合搜索来做决策。

放在这张课程地图里，AlphaGo / AlphaZero 更靠后：

```text
known model + planning/search + self-play RL + neural network policy/value
```

## 以后读别人 RL 方法时，先问这些问题

看到一篇论文、一段代码、一个项目时，可以先按下面顺序拆解：

```text
1. state 是什么？
2. action 是什么？
3. reward 是什么？
4. episode 怎么开始和结束？
5. 环境模型是否已知？
6. 数据是在线交互、仿真、自博弈，还是历史日志？
7. 它学的是 V、Q、policy，还是 model？
8. 它用表格、线性模型，还是神经网络表示？
9. 它如何探索？
10. 它的 update target 或 loss 是什么？
```

如果这十个问题能回答清楚，大部分 RL 方法就不会只剩下公式名。

## 当前阶段最重要的地图

现在先记住这一版：

```text
Bandit
  学：action value、探索/利用
  问：不知道哪个动作好，如何边试边选？

GridWorld DP
  学：state、policy、V(s)、planning
  问：知道环境规则，如何推算长期价值并改进策略？

Q-learning
  学：Q(s,a)、TD target、model-free
  问：不知道完整环境规则，如何从经验中学习动作价值？

DQN
  学：用神经网络近似 Q(s,a)
  问：状态太大，表格放不下怎么办？

Policy Gradient
  学：直接优化 policy
  问：不通过 Q 表，能不能直接学怎么行动？

Actor-Critic / PPO
  学：policy 和 value 配合训练
  问：如何让深度 RL 在复杂任务中更稳定？
```

这就是目前课程的主线：

```text
从小表格开始
先理解决策和价值
再理解从经验学习
最后用神经网络扩展到复杂问题
```

第 0 课不是要求你现在掌握所有后续算法。它只是提供一个导航：当你学到新内容时，知道它是在补哪一块，而不是把每一课当成孤立知识点。
