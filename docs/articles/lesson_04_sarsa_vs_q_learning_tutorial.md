# 第四课：SARSA vs Q-learning，看懂 on-policy 和 off-policy

第三课我们学了 Q-learning。它能在不知道完整环境模型的情况下，通过一条条经验更新动作价值 `Q(s,a)`。

但第三课里留下了一个重要术语：

```text
off-policy
```

只用一句话解释，容易变成背概念：

```text
Q-learning 的行为策略是 epsilon-greedy，目标策略是 greedy，所以它是 off-policy。
```

这句话本身没错，但如果没有对照，很难真正理解。

第四课就用 SARSA 做对照。SARSA 和 Q-learning 非常像：

```text
都在 GridWorld 里训练
都学习 Q(s,a)
都使用 epsilon-greedy 探索
都用 TD error 更新旧估计
```

它们的关键差别只有一个：

```text
SARSA 的 target 使用下一步实际会采取的 action。
Q-learning 的 target 使用下一状态里最大的 Q 值。
```

这句话还可以展开成两个更具体的学习重点：

| 对比点 | Q-learning | SARSA |
| --- | --- | --- |
| 学习对象 | `Q(s,a)` | `Q(s,a)` |
| 行动方式 | 可以用 epsilon-greedy 探索 | 可以用 epsilon-greedy 探索 |
| 更新外壳 | `old_q + alpha * (target - old_q)` | `old_q + alpha * (target - old_q)` |
| 核心差异 | `target = reward + gamma * max_a Q(next_state, a)` | `target = reward + gamma * Q(next_state, next_action)` |
| 连带实现差异 | 不需要知道下一步实际动作 | 需要先选出 `next_action` |

所以，SARSA 和 Q-learning 不是两个完全无关的算法。它们的骨架非常接近，真正改变学习性质的是 `target`。而为了计算 SARSA 的 `target`，代码必须额外拿到 `next_action`。

这一课的目标不是多记一个算法名，而是让你真正看懂：

```text
on-policy 和 off-policy 到底差在哪里。
```

## 1. 本课问题

地图仍然使用前两课的 3x4 GridWorld：

```text
(0,0)  (0,1)  (0,2)  (0,3)=goal +1
(1,0)  WALL   (1,2)  (1,3)=pit  -1
(2,0)  (2,1)  (2,2)  (2,3)
```

起点是：

```text
start = (2,0)
```

动作是：

```text
U/R/D/L = 上/右/下/左
```

奖励是：

```text
到 goal 得到 +1
到 pit 得到 -1
普通走一步得到 -0.04
```

本课的问题是：

```text
如果两个算法都学习 Q(s,a)，也都用 epsilon-greedy 探索，
为什么一个叫 on-policy，另一个叫 off-policy？
```

这一课会始终围绕两个问题展开：

```text
1. target 是如何计算出来的？
2. 为了计算 target，算法需要哪些信息？
```

## 2. 运行实验

在项目根目录运行：

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

Main comparison:
  SARSA target      : reward + gamma * Q(next_state, next_action actually chosen)
  Q-learning target : reward + gamma * max_a Q(next_state, a)
```

## 3. 先看输出

输出分成几块。

第一块是 `config`，说明本次实验参数。

```text
algorithm = both
```

表示同一次运行里会分别训练 SARSA 和 Q-learning。

训练日志里：

```text
sarsa      episode  1000 | avg return last 100: ...
q_learning episode  1000 | avg return last 100: ...
```

分别表示两个算法训练到第 1000 局时，最近 100 局的表现。

几个指标含义和第三课一样：

| 字段 | 含义 |
| --- | --- |
| `avg return last 100` | 最近 100 局平均累计奖励 |
| `avg steps` | 最近 100 局平均走了多少步 |
| `success rate` | 最近 100 局到达 goal 的比例 |

后面还会打印每个算法的：

```text
best Q value by state
greedy policy
```

`best Q value by state` 是：

```text
max_a Q(s,a)
```

`greedy policy` 是：

```text
训练后在每个状态选择 Q 值最大的动作。
```

注意，不要把某一次输出里的高低当作绝对结论。这里的重点不是证明“谁永远更好”，而是看懂两者更新目标的差异。

## 4. 从第三课复习 Q-learning

第三课的核心更新是：

```text
old_q = Q(s,a)
target = r + gamma * max_a Q(s',a)
td_error = target - old_q
new_q = old_q + alpha * td_error
```

其中最关键的是：

```text
max_a Q(s',a)
```

它表示：

```text
到了 next_state 之后，假设下一步会选择当前 Q 值最大的动作。
```

但是训练时，agent 实际行动并不是永远 greedy，而是 epsilon-greedy：

```text
大多数时候选当前最优动作
少数时候随机探索
```

所以 Q-learning 的特点是：

```text
实际行动会探索
学习目标却朝 greedy 策略靠近
```

这就是 off-policy 的基础。

## 5. SARSA 多看了什么

SARSA 这个名字来自一条经验中的五个元素：

```text
S, A, R, S', A'
```

也就是：

```text
state
action
reward
next_state
next_action
```

Q-learning 用的是：

```text
state, action, reward, next_state
```

SARSA 比它多用了：

```text
next_action
```

这个 `next_action` 不是理论上最优的动作，而是 agent 到达 `next_state` 后，按当前行为策略实际选出来的动作。

如果当前行为策略是 epsilon-greedy，那么 `next_action` 也来自 epsilon-greedy。

## 6. SARSA 的更新目标

SARSA 的更新是：

```text
target = r + gamma * Q(s', a')
```

这里：

```text
s' = next_state
a' = next_action actually chosen
```

也就是：

```text
我这一步到达了 next_state，
下一步我实际准备做 next_action，
那我就用 Q(next_state, next_action) 来估计未来。
```

这和 Q-learning 的差别是：

```text
Q-learning：未来按最优动作估计
SARSA：未来按实际会采取的动作估计
```

## 7. 查看单步更新

运行：

```bash
python lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py --episodes 3 --debug-episodes 1 --log-every 0 --max-steps 20
```

你会看到：

```text
sarsa episode 1:
  step  state   action  reward  next    next action  old Q   target  td err  new Q
     1  (2, 0)    L     -0.04 (2, 0)       L         0.00   -0.04   -0.04   -0.01
```

这一行表示：

```text
当前在 (2,0)
实际选择 L
得到 reward = -0.04
到达 next_state = (2,0)
下一步实际会选择 next_action = L
old Q 是 0.00
target 是 -0.04 + gamma * Q((2,0), L)
new Q 更新成 -0.01
```

同样看 Q-learning：

```text
q_learning episode 1:
  step  state   action  reward  next    next action  old Q   target  td err  new Q
     1  (2, 0)    L     -0.04 (2, 0)       -         0.00   -0.04   -0.04   -0.01
```

这里 `next action` 是 `-`，不是因为 Q-learning 没有下一步动作，而是因为它的更新目标不需要知道下一步实际动作。

Q-learning 只需要：

```text
max_a Q(next_state, a)
```

也就是下一状态下当前看起来最好的动作价值。

## 8. 代码执行顺序

源码在：

```text
lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py
```

先看执行流：

```text
main()
  -> 创建 GridWorld
  -> 根据 --algorithm 决定运行哪些算法
  -> train()
      -> reset 到起点
      -> choose_action() 选当前动作
      -> env.step() 得到 next_state/reward/done
      -> 计算 target
      -> 计算 td_error
      -> 更新 q[state][action]
  -> 打印结果
```

这节课的源码大部分和第三课相同。你应该重点看 `train()` 里的 target 计算。

## 9. 两种算法共用的部分

两者都创建 Q 表：

```python
q = make_q_table(env)
```

`q[state][action]` 对应：

```text
Q(s,a)
```

两者都使用 epsilon-greedy 选择动作：

```python
action = choose_action(q, state, epsilon, rng)
```

也就是说：

```text
SARSA 和 Q-learning 都会探索。
```

所以不要误解成：

```text
SARSA 会探索，Q-learning 不探索。
```

真正区别在 target。

## 10. SARSA 的代码

SARSA 先选下一步实际动作：

```python
if algorithm == "sarsa" and not done:
    next_action = choose_action(q, next_state, epsilon, rng)
```

然后用这个动作计算 target：

```python
target = reward + gamma * q[next_state][next_action]
```

对应：

```text
r + gamma * Q(s', a')
```

因为 `next_action` 来自当前行为策略，所以 SARSA 学的是当前行为策略的价值。

## 11. Q-learning 的代码

Q-learning 不看下一步实际动作：

```python
target = reward + gamma * max(q[next_state].values())
```

对应：

```text
r + gamma * max_a Q(s', a)
```

它的意思是：

```text
到了 next_state 以后，按当前 Q 表里最好的动作估计未来。
```

所以，Q-learning 即使训练时用 epsilon-greedy 行动，它的学习目标仍然是 greedy policy。

## 12. 两者共用的 TD 更新外壳

target 算出来后，两者都一样：

```python
td_error = target - old_q
q[state][action] = old_q + alpha * td_error
```

这还是前面反复见到的模式：

```text
new estimate = old estimate + step size * error
```

第四课真正要记住的是：

| 算法 | target | 学到的策略价值 |
| --- | --- | --- |
| SARSA | `r + gamma * Q(s', a')` | 当前行为策略 |
| Q-learning | `r + gamma * max_a Q(s', a)` | greedy 目标策略 |

## 13. 为什么 SARSA 是 on-policy

`policy` 是 agent 选择动作的方式。

SARSA 行动时使用 epsilon-greedy。

SARSA 学习时，target 里的 `next_action` 也来自 epsilon-greedy。

所以它是：

```text
用什么策略行动，就评估和改进什么策略。
```

这就是 on-policy。

## 14. 为什么 Q-learning 是 off-policy

Q-learning 行动时可以使用 epsilon-greedy。

但 Q-learning 学习时使用：

```text
max_a Q(s', a)
```

这相当于在学习 greedy policy 的价值。

所以它是：

```text
用 epsilon-greedy 收集经验，
却朝 greedy policy 学习。
```

这就是 off-policy。

## 15. 为什么这个区别有实际意义

这一课不是为了证明 SARSA 一定比 Q-learning 好，也不是为了证明 Q-learning 一定比 SARSA 好。

如果只看这个小 GridWorld，你可能会觉得：

```text
两者结果差不多。
```

这是正常的。原因是：

```text
地图很小；
环境默认是确定性的；
goal 和 pit 很容易区分；
两个算法最终都能学到接近可用的 Q 值；
最后打印的是 greedy policy，会隐藏训练时探索策略的差异。
```

所以，本课的核心目的不是比较两个算法在这个小地图上的分数高低，而是建立一个以后读 RL 方法时必须用到的判断维度：

```text
数据是从哪个 policy 收集来的？
算法更新时，又是在学习哪个 policy？
```

这个问题在更大的 RL 系统里会影响：

```text
能不能复用旧数据？
能不能使用经验回放 replay buffer？
训练时探索是否会被计入风险？
学到的是当前实际执行策略，还是另一个更贪心的目标策略？
训练稳定性和样本效率如何？
```

如果环境很安全，探索动作代价不高，Q-learning 朝 greedy policy 学习可能很有效。

但如果探索动作本身会带来额外风险，SARSA 会更直接地把这种风险反映进价值估计里。

原因是：

```text
SARSA 的 target 里包含 next_action。
如果 next_action 可能因为探索而走向危险区域，
这个风险会进入 Q(s,a) 的估计。
```

Q-learning 则更像在说：

```text
虽然我训练时会探索，
但我估计未来时先假设自己会走当前最优动作。
```

这不是绝对优劣，而是学习目标不同。

### on-policy 的优缺点

on-policy 方法学习的是当前实际行为策略。

优点：

```text
学习目标和实际行为一致；
更直接反映探索动作带来的风险；
概念上更稳妥，适合关心“当前策略实际表现”的任务。
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

### off-policy 的优缺点

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

### 怎么选择

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

## 16. 参数实验

### 实验 1：只运行 SARSA

```bash
python lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py --algorithm sarsa
```

观察：

```text
sarsa summary
sarsa best Q value by state
sarsa greedy policy
```

目的：确认 SARSA 不是一个全新环境，而是在同一个 GridWorld 中学习 `Q(s,a)`。

### 实验 2：只运行 Q-learning

```bash
python lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py --algorithm q_learning
```

目的：对照第三课，确认 Q-learning 的核心逻辑仍然是：

```text
reward + gamma * max_a Q(next_state, a)
```

### 实验 3：增大 epsilon

```bash
python lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py --epsilon 0.5 --log-every 1000
```

观察：

```text
SARSA 和 Q-learning 的 Q 值差异是否更明显？
最终 greedy policy 是否不同？
```

原因是：`epsilon` 越大，实际行为中随机动作越多。SARSA 的 target 会考虑这些实际动作，Q-learning 的 target 仍然看 greedy 最大值。

### 实验 4：加入环境随机性

```bash
python lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py --slip-probability 0.1 --log-every 1000
```

观察：

```text
success rate 是否下降？
avg return 是否更波动？
靠近 pit 的路径是否变得不稳定？
```

这能帮助你理解：算法差异不仅来自公式，也会被环境风险放大或缩小。

### 实验 5：减少训练局数

```bash
python lessons/04_sarsa_vs_q_learning/sarsa_vs_q_learning.py --episodes 200 --log-every 100
```

观察：

```text
训练数据不够时，最终 greedy policy 是否更容易出现奇怪动作？
```

这能提醒你：不要只用算法名判断效果，训练样本量也很关键。

## 17. 本课总结

这一课可以压缩成一句话：

```text
SARSA 和 Q-learning 都学习 Q(s,a)，区别在于 target 如何估计未来。
```

更具体：

```text
SARSA:
  target = reward + gamma * Q(next_state, next_action actually chosen)
  学当前行为策略的价值
  on-policy

Q-learning:
  target = reward + gamma * max_a Q(next_state, a)
  学 greedy 目标策略的价值
  off-policy
```

如果你能看懂这一点，后面再遇到 DQN、Actor-Critic、PPO 时，就能更快判断：

```text
这个方法的数据是怎么来的？
它实际执行的策略是什么？
它更新时假设的目标策略又是什么？
```

这比单纯记住算法名字更重要。

## 18. 进入下一课前的检查

你应该能用自己的话回答：

1. SARSA 里的五个字母分别是什么？
2. SARSA 和 Q-learning 是否都学习 `Q(s,a)`？
3. SARSA 为什么需要 `next_action`？
4. Q-learning 为什么不需要 `next_action`？
5. on-policy 中的 policy 指什么？
6. off-policy 中的“off”到底错开了什么？
7. 如果 `epsilon` 变大，为什么两种算法的差异可能变大？

如果这些问题能回答清楚，表格 Q 方法里“用一步经验更新动作价值，并通过动作价值改进策略”的主干就基本打通了。

以后你可能会看到一个术语：

```text
TD control
```

现在可以先把它理解成：

```text
用 TD error 更新 Q(s,a)，并通过 Q 值改进行动策略的方法。
```

SARSA 和 Q-learning 都属于这个大类。但这一课不需要先记分类名，重点是看懂两种 `target` 的不同。

下一步就可以开始问一个更现实的问题：

```text
如果 state 太多，Q 表放不下怎么办？
```
