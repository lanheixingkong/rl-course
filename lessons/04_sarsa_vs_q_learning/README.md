# Lesson 04: SARSA vs Q-learning

本课目标：在第三课 Q-learning 的基础上，真正看懂 `on-policy` 和 `off-policy` 的区别。

第三课已经知道：

```text
Q-learning 训练时用 epsilon-greedy 探索，
但更新目标使用 next_state 里最大的 Q 值。
```

这节课加入 SARSA。它和 Q-learning 很像，也学习 `Q(s,a)`，也使用经验：

```text
state, action, reward, next_state
```

但它多看一个东西：

```text
next_action
```

也就是下一步实际会采取的动作。

先给出本课最重要的结论，方便和第三课对比：

| 对比点 | 第三课 Q-learning | 第四课 SARSA |
| --- | --- | --- |
| 学习对象 | `Q(s,a)` | `Q(s,a)` |
| 行动方式 | 可以用 epsilon-greedy 探索 | 可以用 epsilon-greedy 探索 |
| 更新外壳 | `old_q + alpha * (target - old_q)` | `old_q + alpha * (target - old_q)` |
| 核心差异 | `target = reward + gamma * max_a Q(next_state, a)` | `target = reward + gamma * Q(next_state, next_action)` |
| 连带实现差异 | 不需要知道下一步实际动作 | 必须先选出 `next_action` |

所以，本课不是在换一个完全不同的算法，而是在同一个 Q 值更新流程里比较：

```text
target 怎么算？
为了算这个 target，代码需要哪些变量？
```

## 这一课解决什么问题

第三课里我们说 Q-learning 是 `off-policy`，因为：

```text
行为策略：epsilon-greedy，训练时会探索
学习目标：greedy，总是假设下一步选当前 Q 值最大的动作
```

但这句话容易停留在术语层面。

本课用 SARSA 做对照：

```text
SARSA：用下一步实际会采取的 next_action 来更新
Q-learning：用下一状态里最大的 Q 值来更新
```

两者都在同一个 GridWorld 里训练。这样可以把注意力集中到两个差异点：

```text
1. target 到底是怎么算出来的？
2. 为了计算 target，SARSA 为什么需要 next_action，而 Q-learning 不需要？
```

地图仍然是：

```text
(0,0)  (0,1)  (0,2)  (0,3)=goal +1
(1,0)  WALL   (1,2)  (1,3)=pit  -1
(2,0)  (2,1)  (2,2)  (2,3)
```

起点：

```text
start = (2,0)
```

动作：

| 符号 | 含义 |
| --- | --- |
| `U` | up，向上 |
| `R` | right，向右 |
| `D` | down，向下 |
| `L` | left，向左 |

## 运行

```bash
python lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py
```

你会看到类似输出：

```text
SARSA vs Q-learning GridWorld config:
  algorithm        : both
  episodes         : 5000
  max_steps        : 100
  alpha            : 0.2
  gamma            : 0.95
  epsilon          : 0.2
  step_reward      : -0.04
  goal_reward      : 1.0
  pit_reward       : -1.0
  slip_probability : 0.0
  seed             : 0

sarsa      episode  1000 | avg return last 100:  0.781 | avg steps:   6.5 | success rate:  1.00
...
q_learning episode  5000 | avg return last 100:  0.696 | avg steps:   6.1 | success rate:  0.95

sarsa summary:
  avg return over last 100: 0.789
  avg steps over last 100 : 6.3
  success rate last 100   : 1.00

q_learning summary:
  avg return over last 100: 0.696
  avg steps over last 100 : 6.1
  success rate last 100   : 0.95

Main comparison:
  SARSA target      : reward + gamma * Q(next_state, next_action actually chosen)
  Q-learning target : reward + gamma * max_a Q(next_state, a)
```

## 怎么读输出

### config 是什么

| 参数 | 含义 |
| --- | --- |
| `algorithm` | 运行 `sarsa`、`q_learning`，还是两个都运行 |
| `episodes` | 训练多少局 |
| `max_steps` | 每局最多走多少步 |
| `alpha` | 学习率，新经验影响旧 Q 值的幅度 |
| `gamma` | 折扣因子，未来奖励的重要程度 |
| `epsilon` | 探索概率，有多大概率随机选动作 |
| `step_reward` | 普通走一步的奖励，默认是小惩罚 |
| `slip_probability` | 动作打滑概率，模拟环境随机性 |

### 训练日志是什么

```text
sarsa      episode  1000 | avg return last 100:  0.781 | avg steps:   6.5 | success rate:  1.00
q_learning episode  1000 | avg return last 100:  0.800 | avg steps:   6.0 | success rate:  1.00
```

意思是：

| 字段 | 含义 |
| --- | --- |
| `sarsa` / `q_learning` | 当前是哪种算法的训练日志 |
| `episode 1000` | 已训练到第 1000 局 |
| `avg return last 100` | 最近 100 局平均累计奖励 |
| `avg steps` | 最近 100 局平均走了多少步 |
| `success rate` | 最近 100 局到达 goal 的比例 |

注意：两种算法的日志不能只看单次高低。因为训练过程中有随机探索，短窗口内的 `avg return` 会波动。

### Best Q value 和 greedy policy 是什么

这部分和第三课一样。

`Best Q value by state` 表示每个状态下最好的动作价值：

```text
max_a Q(s,a)
```

`greedy policy` 表示训练后如果不再探索，而是在每个状态都选 Q 值最大的动作，会得到什么策略。

这里要注意：SARSA 训练时学的是“包含探索的行为策略”的价值，但最后打印的仍然是 greedy policy。这样做是为了方便和 Q-learning 对比最终学到的 Q 表。

## 查看单步更新

运行：

```bash
python lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py --episodes 3 --debug-episodes 1 --log-every 0 --max-steps 20
```

你会看到类似：

```text
sarsa episode 1:
  step  state   action  reward  next    next action  old Q   target  td err  new Q
     1  (2, 0)    L     -0.04 (2, 0)       L         0.00   -0.04   -0.04   -0.01
...

q_learning episode 1:
  step  state   action  reward  next    next action  old Q   target  td err  new Q
     1  (2, 0)    L     -0.04 (2, 0)       -         0.00   -0.04   -0.04   -0.01
```

重点看 `next action`：

```text
SARSA 有 next action，因为它要用下一步实际会采取的动作更新。
Q-learning 的 next action 是 -，因为它不需要下一步实际动作，只需要 next_state 里最大的 Q 值。
```

## 代码和原理对照

先看源码位置：

```text
lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py
```

执行顺序是：

```text
main()
  -> 创建 GridWorld
  -> 根据 --algorithm 决定运行 sarsa、q_learning 或 both
  -> train()
      -> reset 到起点
      -> choose_action() 选择动作
      -> env.step() 得到 next_state/reward/done
      -> 根据算法计算 target
      -> 更新 q[state][action]
  -> 打印 summary、best Q value、greedy policy
```

### 1. 两种算法共用 Q 表

```python
q = make_q_table(env)
```

`q[state][action]` 对应：

```text
Q(s,a)
```

也就是：

```text
在 state s 做 action a，长期预计能得到多少累计奖励。
```

### 2. 两种算法共用 epsilon-greedy 行动

```python
action = choose_action(q, state, epsilon, rng)
```

`choose_action()` 的逻辑和第三课一样：

```text
以 epsilon 的概率随机探索；
否则选择当前 Q 值最大的动作。
```

所以，SARSA 和 Q-learning 的区别不是“谁探索、谁不探索”。默认情况下，两者训练时都会探索。

### 3. SARSA 的 target

源码中对应这段：

```python
if algorithm == "sarsa" and not done:
    next_action = choose_action(q, next_state, epsilon, rng)
```

这表示：SARSA 到达 `next_state` 后，会先按当前行为策略选出下一步实际动作 `next_action`。

然后：

```python
target = reward + gamma * q[next_state][next_action]
```

对应：

```text
target = r + gamma * Q(s', a')
```

这里的 `a'` 不是“理论最优动作”，而是下一步实际会采取的动作。因为它包含 epsilon-greedy 探索，所以 SARSA 学到的是当前行为策略的价值。

这就是 `on-policy`。

### 4. Q-learning 的 target

Q-learning 不使用 `next_action`。

源码是：

```python
target = reward + gamma * max(q[next_state].values())
```

对应：

```text
target = r + gamma * max_a Q(s', a)
```

也就是说，即使训练时当前 agent 仍然会随机探索，Q-learning 的学习目标也是假设下一步会选择当前看来最好的动作。

这就是 `off-policy`。

### 5. 两者共用同一个更新外壳

```python
td_error = target - old_q
q[state][action] = old_q + alpha * td_error
```

这和第三课一样：

```text
new estimate = old estimate + step size * error
```

真正不同的是 `target`：

| 算法 | target |
| --- | --- |
| SARSA | `reward + gamma * Q(next_state, next_action)` |
| Q-learning | `reward + gamma * max_a Q(next_state, a)` |

## 为什么 SARSA 是 on-policy

`policy` 可以理解为“agent 实际怎么选动作”。

SARSA 的 target 使用的是：

```text
下一步实际会采取的 action
```

如果当前行为策略是 epsilon-greedy，那么 `next_action` 也来自 epsilon-greedy。

因此 SARSA 评估和改进的是同一个策略：

```text
用 epsilon-greedy 行动
也用 epsilon-greedy 的下一步动作来学习
```

这就是 on-policy。

## 为什么 Q-learning 是 off-policy

Q-learning 行动时可以用 epsilon-greedy：

```text
有时随机探索
```

但它学习时使用：

```text
next_state 中最大的 Q 值
```

也就是学习目标朝 greedy policy 靠近。

因此它是：

```text
行为策略：epsilon-greedy
目标策略：greedy
```

行为策略和目标策略不完全一样，所以叫 off-policy。

## 为什么要区分 on-policy 和 off-policy

这一课的目的不是证明 SARSA 一定比 Q-learning 好，或者 Q-learning 一定比 SARSA 好。

在这个小 GridWorld 里，你可能看不到特别大的结果差异，这是正常的。原因有几个：

```text
地图很小；
环境默认是确定性的；
goal 和 pit 很容易区分；
两个算法最终都能学到接近可用的 Q 值；
最后打印的是 greedy policy，会隐藏训练时探索策略的差异。
```

所以，本课真正要训练的是一个判断问题：

```text
数据是从哪个 policy 收集来的？
算法更新时，又是在学习哪个 policy？
```

这件事在小 GridWorld 里看起来只是术语差异，但在更大的 RL 系统里会直接影响：

```text
能不能复用旧数据？
能不能使用经验回放 replay buffer？
训练时探索是否会被计入风险？
学到的是当前实际执行策略，还是另一个更贪心的目标策略？
训练稳定性和样本效率如何？
```

### on-policy 的特点

on-policy 方法学习的是当前实际行为策略。

优点：

```text
学习目标和实际行为一致；
更直接反映探索动作带来的风险；
概念上更稳妥，尤其适合关心“当前策略实际表现”的任务。
```

缺点：

```text
通常不能随便复用很旧的数据；
如果策略更新了，旧数据可能不再代表当前策略；
样本效率往往较低，需要不断用当前策略收集新经验。
```

典型适用场景：

```text
需要评估和改进当前正在执行的策略；
探索动作本身有风险，不能假装未来总会选最优动作；
后续学习 Policy Gradient、Actor-Critic、PPO 时，需要理解这一类思想。
```

### off-policy 的特点

off-policy 方法允许“收集数据的策略”和“学习目标策略”不同。

优点：

```text
可以用探索策略收集数据，同时学习更贪心的目标策略；
更容易复用历史经验；
可以配合 replay buffer，提高样本利用率；
后续 DQN 就是重要的 off-policy 深度 RL 方法。
```

缺点：

```text
学习目标和实际行为不一致，可能更乐观；
在复杂函数逼近场景中更容易不稳定；
如果历史数据和目标策略差太远，会带来分布不匹配问题。
```

典型适用场景：

```text
想从探索数据中学习一个更优的 greedy 策略；
希望复用过去收集到的经验；
使用经验回放或离线数据；
后续学习 Q-learning、DQN、部分离线 RL 方法时，需要理解这一类思想。
```

### 简单选择原则

先用这个粗略判断：

| 问题 | 更偏向 |
| --- | --- |
| 我关心当前实际执行策略的表现 | on-policy |
| 探索动作本身有明显风险 | on-policy 更容易把风险计入价值 |
| 我想从探索数据中学习更贪心的目标策略 | off-policy |
| 我希望复用旧经验或 replay buffer | off-policy |
| 我只有历史日志，不能重新在线采样 | off-policy / offline RL 方向 |
| 我要学习 PPO 这类策略优化方法 | 先理解 on-policy |
| 我要学习 DQN 这类价值方法 | 先理解 off-policy |

这张表不是绝对规则。真实项目里还要看安全性、数据来源、环境是否可模拟、状态规模、函数逼近稳定性等因素。

## 参数实验

### 实验 1：只运行 SARSA

```bash
python lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py --algorithm sarsa
```

看：

```text
sarsa summary
sarsa best Q value by state
sarsa greedy policy
```

目的：先单独确认 SARSA 也能学到可用策略。

### 实验 2：只运行 Q-learning

```bash
python lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py --algorithm q_learning
```

目的：对照第三课，确认这里的 Q-learning 逻辑没有变，只是放进了比较框架。

### 实验 3：增大 epsilon

```bash
python lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py --epsilon 0.5 --log-every 1000
```

要观察的问题：

```text
探索变多后，SARSA 和 Q-learning 的 Q 值差异是否更明显？
```

原因是：SARSA 的 target 会考虑“下一步也可能探索”，所以探索概率越大，它学到的价值越会反映探索带来的风险和绕路成本。

### 实验 4：加入打滑

```bash
python lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py --slip-probability 0.1 --log-every 1000
```

要观察的问题：

```text
环境更随机后，两种算法的 success rate、avg return、greedy policy 是否变化？
```

打滑让动作结果不稳定。靠近 pit 的路径可能更危险，因此策略可能更保守。

### 实验 5：减少训练局数

```bash
python lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py --episodes 200 --log-every 100
```

要观察的问题：

```text
经验不够时，两种算法的 greedy policy 是否更不稳定？
```

这能提醒你：不要只看最终算法名，还要看训练数据是否足够。

## 本课过关问题

进入下一课前，你需要能回答：

1. SARSA 和 Q-learning 学习的对象是不是都是 `Q(s,a)`？
2. SARSA 的 `target` 怎么算？
3. Q-learning 的 `target` 怎么算？
4. 为什么 SARSA 的 debug 输出里有 `next action`？
5. 为什么 Q-learning 不需要 `next action`？
6. `on-policy` 中的 policy 指的是什么？
7. Q-learning 为什么训练时探索，但仍然叫 off-policy？
8. 当 `epsilon` 更大时，为什么 SARSA 和 Q-learning 的差异可能更明显？

## 这一课和下一课的关系

这一课补齐了一个重要比较：

```text
SARSA：用实际 next_action 计算 target，所以是 on-policy
Q-learning：用 next_state 的最大 Q 值计算 target，所以是 off-policy
```

如果以后看到 `TD control` 这个术语，可以先把它理解成：

```text
用 TD error 更新 Q(s,a)，并通过 Q 值改进行动策略的方法。
```

SARSA 和 Q-learning 都属于这个大类。但在本课里，先不用记这个分类名，重点仍然是看懂两种 `target` 的差异。

学完它以后，你再看后面的 DQN、Policy Gradient、Actor-Critic，会更容易判断一个方法到底是在学：

```text
行为策略本身
还是另一个目标策略
```

下一步会开始处理一个新问题：表格 Q 方法只能用于很小的状态空间。如果状态很多，甚至是连续状态，就不能再给每个 `(state, action)` 存一个表格值，需要进入函数逼近。
