# 第三课：Q-learning，在不知道环境模型时学习行动价值

前两课我们已经学了两个重要阶段。

第一课 Bandit 只有动作，没有状态。它让我们理解了探索和利用。

第二课 GridWorld Dynamic Programming 有状态、有动作、有奖励、有策略，但它有一个很强的前提：环境模型已知。也就是说，agent 可以提前知道：

```text
在 state 做 action，会到哪个 next_state，得到多少 reward
```

更具体一点，第二课的算法可以在不真正走路的情况下，直接问完整规则：

```text
如果在 (2,0) 做 U，会到 (1,0)，reward = -0.04
如果在 (2,0) 做 R，会到 (2,1)，reward = -0.04
如果在 (2,0) 做 D，会撞边界，留在 (2,0)，reward = -0.04
如果在 (2,0) 做 L，会撞边界，留在 (2,0)，reward = -0.04
```

然后它会对所有状态、所有动作做这种“脑内模拟”，再推算 value 和 policy。

第三课开始，我们去掉这个学习方式。agent 不再提前使用完整环境模型规划，而是通过自己走出来的一条条经验学习。

Q-learning 的学习方式更像：

```text
我现在在 (2,0)
我这一步实际选了 R
环境告诉我：到了 (2,1)，reward = -0.04
那我只更新 Q((2,0), R)
```

它每次只使用刚刚发生的一条经验：

```text
state, action, reward, next_state
```

可以先用这张表区分第二课和第三课：

| 问题 | 第二课 Dynamic Programming | 第三课 Q-learning |
| --- | --- | --- |
| 是否提前使用完整环境模型 | 是 | 否 |
| 一次更新用什么 | 已知规则推算出的 `next_state/reward` | 实际走出来的 `(state, action, reward, next_state)` |
| 更新对象 | `V(s)`，状态价值 | `Q(s,a)`，动作价值 |
| 是否遍历所有状态 | 是 | 训练中按经验遇到哪些就更新哪些 |
| 像什么 | 拿着地图和规则做全局推演 | 边走边记经验，逐步修正判断 |

这就是 Q-learning 的入口。

## 1. 本课问题

地图仍然是同一个 3x4 GridWorld：

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

这一课的问题是：

```text
如果 agent 不能提前用完整环境模型做 planning，
只通过每次行动后看到的 reward 和 next_state，
能不能学到一个好策略？
```

## 2. 运行实验

在项目根目录运行：

```bash
python lessons/03_q_learning/q_learning_gridworld.py
```

你会看到类似：

```text
Q-learning GridWorld config:
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

episode   500 | avg return last 100:  0.779 | avg steps:   6.0 | success rate:  0.99
...
episode  5000 | avg return last 100:  0.696 | avg steps:   6.1 | success rate:  0.95

Training summary:
  avg return over last 100: 0.696
  avg steps over last 100 : 6.1
  success rate last 100   : 0.95

Best Q value by state:
  0.82   0.91   1.00   1.00
  0.74   WALL    0.91  -1.00
  0.67   0.74   0.82   0.74

Greedy policy:
  R    R    R    +1
  U  WALL   U    -1
  U    R    U    L
```

## 3. 先看输出

`episodes` 表示训练多少局。每一局从起点 `(2,0)` 开始，到达 goal、pit，或者达到 `max_steps` 后结束。

训练日志里的：

```text
avg return last 100
```

表示最近 100 局的平均累计奖励。

```text
avg steps
```

表示最近 100 局平均用了多少步。

```text
success rate
```

表示最近 100 局到达 goal 的比例。

注意，训练日志不一定单调变好。因为训练时 `epsilon=0.2`，agent 仍然有 20% 的概率随机探索。探索动作可能绕路，也可能掉坑，所以平均回报会波动。

`Best Q value by state` 表示每个状态下最好的 Q 值：

```text
max_a Q(s,a)
```

`Greedy policy` 表示训练结束后，在每个状态选择 Q 值最大的动作。

## 4. 从 V(s) 到 Q(s,a)

第二课的核心对象是：

```text
V(s): 状态 s 的价值
```

它回答：

```text
从这个状态出发，长期大概能拿多少 reward？
```

第三课的核心对象是：

```text
Q(s,a): 在状态 s 做动作 a 的价值
```

它回答：

```text
如果我在这个状态先做这个动作，然后继续好好走，长期大概能拿多少 reward？
```

所以 Q 比 V 更具体。

```text
V(s) 只评价状态
Q(s,a) 评价状态里的某个动作
```

有了 Q 值以后，策略就很直接：

```text
在每个状态，选择 Q 值最大的动作
```

这就是最终打印出来的 `Greedy policy`。

## 5. 查看一条经验如何更新 Q 值

运行：

```bash
python lessons/03_q_learning/q_learning_gridworld.py --episodes 5 --debug-episodes 1 --log-every 0
```

你会看到：

```text
Episode 1:
  step  state   action  reward  next    old Q   target  td err  new Q
     1  (2, 0)    L     -0.04 (2, 0)     0.00   -0.04   -0.04   -0.01
     2  (2, 0)    R     -0.04 (2, 1)     0.00   -0.04   -0.04   -0.01
     3  (2, 1)    D     -0.04 (2, 1)     0.00   -0.04   -0.04   -0.01
     4  (2, 1)    U     -0.04 (2, 1)     0.00   -0.04   -0.04   -0.01
```

这一行就是一条 Q-learning 经验。

| 列 | 含义 |
| --- | --- |
| `state` | 当前状态 |
| `action` | agent 选择的动作 |
| `reward` | 环境返回的即时奖励 |
| `next` | 环境返回的下一状态 |
| `old Q` | 更新前的动作价值 |
| `target` | 这次经验给出的学习目标 |
| `td err` | target 和 old Q 的差距 |
| `new Q` | 更新后的动作价值 |

这说明 Q-learning 不需要一次知道整张地图的所有转移规则。它只需要一条刚发生的经验：

```text
state, action, reward, next_state
```

这里最容易误解的是：step1 的 `new Q = -0.01`，为什么 step2 的 `old Q` 还是 `0.00`？

原因是 Q-learning 存的不是“每个 state 一个值”，而是：

```text
每个 state-action 一份值
```

step1 更新的是：

```text
Q((2,0), L)
```

step2 读取的是：

```text
Q((2,0), R)
```

它们是两个不同格子里的不同动作价值。

可以把 Q 表想成：

```text
状态 (2,0):
  Q((2,0), U) =  0.00
  Q((2,0), R) =  0.00  <- step2 更新这个
  Q((2,0), D) =  0.00
  Q((2,0), L) = -0.01  <- step1 刚更新的是这个
```

所以一个动作的 `new Q` 不会自动变成另一个动作的 `old Q`。

再看 `target`。

`target` 是这次经验给出的学习目标：

```text
target = reward + gamma * next_best
```

在 step1 里，`reward = -0.04`，因为 `(2,0)` 向左撞边界，没有到 goal，也没有到 pit，所以环境给普通走路奖励：

```text
step_reward = -0.04
```

训练刚开始时，所有 Q 值都是 `0.00`，所以：

```text
next_best = 0.00
target = -0.04 + 0.95 * 0.00 = -0.04
```

因此 step1 的 `target = -0.04` 确实和 `step_reward = -0.04` 有直接关系。但它们不是同一个概念：

| 名称 | 含义 |
| --- | --- |
| `step_reward` | 环境规定的普通走一步即时奖励 |
| `reward` | 这一步实际拿到的即时奖励，可能来自 `step_reward`、goal reward 或 pit reward |
| `target` | 用 `reward + gamma * next_best` 算出的学习目标 |

等 Q 值学起来后，`target` 会包含下一状态的未来价值，不再只是 `step_reward`。

## 6. 代码执行顺序

源码在：

```text
lessons/03_q_learning/q_learning_gridworld.py
```

重点读这些部分：

```text
第 21-59 行：GridWorld 环境
第 62-67 行：创建 Q 表
第 70-82 行：epsilon-greedy 选动作
第 85-154 行：Q-learning 训练循环
第 189-230 行：命令行参数和参数检查
第 233-282 行：main() 创建环境、训练、打印结果
```

整体执行顺序是：

```text
main()
  -> 创建 GridWorld
  -> train_q_learning()
       -> 每个 episode 从 start 开始
       -> choose_action() 选择动作
       -> env.step() 得到 next_state, reward, done
       -> 用这条经验更新 Q(state, action)
  -> print_best_values()
  -> print_greedy_policy()
```

## 7. Q 表

代码：

```python
def make_q_table(env: GridWorld) -> dict[State, dict[Action, float]]:
    return {
        state: {action: 0.0 for action in ACTIONS}
        for state in env.states()
        if not env.is_terminal(state)
    }
```

它创建的是：

```text
每个非终止状态 -> 每个动作的 Q 值
```

例如：

```python
q[(2, 0)] = {
    "U": 0.0,
    "R": 0.0,
    "D": 0.0,
    "L": 0.0,
}
```

一开始全是 `0.0`，因为 agent 还没有经验。

## 8. epsilon-greedy

代码：

```python
if rng.random() < epsilon:
    return rng.choice(ACTIONS)
```

意思是：以 `epsilon` 的概率随机探索。

否则：

```python
values = q[state]
best_value = max(values.values())
best_actions = [action for action, value in values.items() if value == best_value]
return rng.choice(best_actions)
```

意思是：选择当前 Q 值最高的动作。如果多个动作并列最高，就随机选一个。

所以 `epsilon=0` 不代表训练早期完全没有随机性。因为初始时四个动作的 Q 值都一样，代码会在并列动作里随机选。

## 9. Q-learning 更新

核心代码：

```python
old_q = q[state][action]
next_best = 0.0 if done else max(q[next_state].values())
target = reward + gamma * next_best
td_error = target - old_q
q[state][action] = old_q + alpha * td_error
```

对应公式：

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

变量对照：

| 公式 | 代码 | 含义 |
| --- | --- | --- |
| `s` | `state` | 当前状态 |
| `a` | `action` | 当前动作 |
| `r` | `reward` | 当前一步奖励 |
| `s'` | `next_state` | 下一状态 |
| `alpha` | `alpha` | 学习率 |
| `gamma` | `gamma` | 折扣因子 |
| `Q(s,a)` | `q[state][action]` | 当前动作价值 |
| `max_a' Q(s',a')` | `max(q[next_state].values())` | 下一状态里最好的动作价值 |

## 10. target 和 TD error

`target` 是这次经验告诉我们的新目标：

```text
target = reward + gamma * next_best
```

它的意思是：

```text
这一步已经拿到 reward；
下一步以后，假设我们会选择当前看来最好的动作。
```

`td_error` 是：

```text
td_error = target - old_q
```

也就是：

```text
新目标 - 旧估计
```

如果 `td_error > 0`，说明这个动作比原来想的更好，Q 值会上调。

如果 `td_error < 0`，说明这个动作比原来想的更差，Q 值会下调。

这和第一课 Bandit 的更新结构是一致的：

```text
new estimate = old estimate + step size * error
```

只是 Q-learning 的 error 里包含了下一状态的未来价值。

## 11. 为什么 Q-learning 不需要 planning

第二课的 policy evaluation 会对所有状态反复计算：

```text
V(s) = r + gamma * V(s')
```

它依赖已知环境模型，能直接问：

```text
如果在 s 做 a，会到哪里？
```

Q-learning 不这样做。它只在真实或模拟交互发生后，用当前这一条经验更新一个 Q 值：

```text
刚才在 s 做了 a
看到了 r 和 s'
所以更新 Q(s,a)
```

这就是 model-free 的核心味道：不显式学习完整环境模型，也不提前枚举所有状态动作后果。

## 12. 为什么 Q-learning 是 off-policy

这一节不是为了多记一个术语，而是解释一个关键问题：

```text
训练时 agent 明明会随机探索，
为什么 Q-learning 最后还能学到一个 greedy 的好策略？
```

Q-learning 同时做了两件看起来不一样的事：

```text
行为上：为了探索，允许自己乱试动作。
学习上：更新 Q 值时，假设未来会选择当前最好的动作。
```

`off-policy` 这个词就是用来描述这种“不按实际行为方式来定义学习目标”的情况。

训练时，agent 的行为策略是 epsilon-greedy：

```text
有时随机探索，有时选当前最好的动作
```

但更新目标里使用：

```python
max(q[next_state].values())
```

这表示：学习时假设未来会选择下一状态里最好的动作。

所以：

```text
行为时：可能随机探索
学习时：朝 greedy 最优目标更新
```

行为策略和学习目标策略不完全一样，这就是 off-policy。

为什么现在要知道它？

第一，它能解释 Q-learning 的核心特性：

```text
可以一边探索，一边学习最终想要的 greedy 策略。
```

第二，后面读其他 RL 方法时经常会遇到这个区分：

| 方法 | 大致特点 |
| --- | --- |
| Q-learning / DQN | off-policy，学习目标偏向 greedy 最优动作 |
| Sarsa | on-policy，学习目标跟实际采取的下一个动作一致 |
| PPO | on-policy 风格，通常用当前策略采样的数据更新当前策略 |

所以这一节的目的不是让你背术语，而是让你看懂：

```text
行为策略和学习目标策略是否是同一个东西？
```

## 13. 参数实验

### 改 episodes

```bash
python lessons/03_q_learning/q_learning_gridworld.py --episodes 100 --log-every 50
```

训练局数少时，有些状态动作可能经验不足。最终策略可能大体正确，但局部动作不稳定。

### 改 epsilon

```bash
python lessons/03_q_learning/q_learning_gridworld.py --episodes 500 --epsilon 0 --log-every 250
python lessons/03_q_learning/q_learning_gridworld.py --episodes 500 --epsilon 0.5 --log-every 250
```

`epsilon=0` 主动探索少，但由于初始 Q 值并列，训练早期仍可能随机选择并列动作。

`epsilon=0.5` 探索多，能覆盖更多动作，但训练期间平均回报可能下降。

### 改 alpha

```bash
python lessons/03_q_learning/q_learning_gridworld.py --episodes 1000 --alpha 0.05 --log-every 500
python lessons/03_q_learning/q_learning_gridworld.py --episodes 1000 --alpha 1.0 --log-every 500
```

`alpha` 小，学习慢但平滑。

`alpha` 大，学习快但更容易受单次经验影响。

### 改 slip_probability

```bash
python lessons/03_q_learning/q_learning_gridworld.py --slip-probability 0.1 --log-every 1000
```

动作有概率变成随机方向后，同一个动作的结果不再完全确定。agent 需要从更多经验里学平均效果。

## 14. 本课你应该带走什么

第一，Q-learning 学的是 `Q(s,a)`，不是只学 `V(s)`。

第二，Q-learning 用一条条经验更新：

```text
state, action, reward, next_state
```

第三，Q-learning 的核心公式不是孤立公式，而是：

```text
旧估计往新目标移动一小步
```

第四，`epsilon` 控制探索，`alpha` 控制学习速度，`gamma` 控制未来重要性。

第五，Q-learning 是 model-free、off-policy 的经典入门算法。

## 15. 进入下一课前的问题

1. `V(s)` 和 `Q(s,a)` 的区别是什么？
2. 为什么 Q-learning 可以从单条经验更新？
3. `target = reward + gamma * next_best` 表达了什么？
4. `td_error` 为正和为负分别意味着什么？
5. `epsilon` 太小和太大分别有什么问题？
6. `alpha` 太小和太大分别有什么问题？
7. Q-learning 为什么不需要提前知道完整环境模型？
8. Q-learning 为什么叫 off-policy？
