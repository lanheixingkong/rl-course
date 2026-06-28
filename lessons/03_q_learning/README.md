# Lesson 03: Q-learning GridWorld

本课目标：从第二课的 `planning` 进入真正的“边走边学”。你会看到 agent 不再提前使用完整环境模型计算最优策略，而是通过一条条经验更新 `Q(s,a)`。

第二课问的是：

```text
如果我知道环境规则，能不能直接算出每个格子的最优动作？
```

第三课问的是：

```text
如果我不知道每个动作长期好不好，只能自己走出来经验，能不能学到策略？
```

## 这一课解决什么问题

地图仍然是同一个 3x4 GridWorld：

```text
(0,0)  (0,1)  (0,2)  (0,3)=goal +1
(1,0)  WALL   (1,2)  (1,3)=pit  -1
(2,0)  (2,1)  (2,2)  (2,3)
```

起点是左下角：

```text
start = (2,0)
```

动作仍然是：

| 符号 | 含义 |
| --- | --- |
| `U` | up，向上 |
| `R` | right，向右 |
| `D` | down，向下 |
| `L` | left，向左 |

这一课和第二课用的是同一个地图，但学习方式不同。

### 第二课：知道完整环境模型，直接推算

“知道完整环境模型”不是一句抽象的话，它具体指：

```text
对任意一个 state 和 action，
agent 都可以提前知道 next_state 和 reward。
```

例如第二课里，算法可以在不真正走路的情况下，直接计算：

```text
如果在 (2,0) 做 U，会到 (1,0)，reward = -0.04
如果在 (2,0) 做 R，会到 (2,1)，reward = -0.04
如果在 (2,0) 做 D，会撞边界，留在 (2,0)，reward = -0.04
如果在 (2,0) 做 L，会撞边界，留在 (2,0)，reward = -0.04
```

也就是说，第二课的算法像是在“脑内模拟整张地图”。它会反复问：

```text
每个格子的每个动作会导致什么结果？
```

然后用这些已知规则推算：

```text
每个 state 的 value 是多少？
每个 state 最后应该选哪个 action？
```

这就是：

```text
知道完整环境模型，直接推算 value 和 policy
```

### 第三课：不提前规划，只用实际经验更新 Q

第三课的 Q-learning 不会在训练前把所有状态、所有动作的后果都推算一遍。

它的学习方式更像：

```text
我现在在 (2,0)
我这一步实际选了 R
环境告诉我：到了 (2,1)，reward = -0.04
那我只更新 Q((2,0), R)
```

也就是说，Q-learning 每次只使用刚刚发生的一条经验：

```text
state, action, reward, next_state
```

它不会同时更新 `(2,0)` 下 `U/R/D/L` 四个动作，也不会立刻遍历整张地图。

### 一句话对比

```text
第二课：我知道规则，所以可以先在脑内把各种可能都算一遍。
第三课：我不先做全局推算，只根据自己实际经历过的一步一步更新。
```

对比表：

| 问题 | 第二课 Dynamic Programming | 第三课 Q-learning |
| --- | --- | --- |
| 是否提前使用完整环境模型 | 是 | 否 |
| 一次更新用什么 | 已知规则推算出的 `next_state/reward` | 实际走出来的 `(state, action, reward, next_state)` |
| 更新对象 | `V(s)`，状态价值 | `Q(s,a)`，动作价值 |
| 是否遍历所有状态 | 是 | 训练中按经验遇到哪些就更新哪些 |
| 像什么 | 拿着地图和规则做全局推演 | 边走边记经验，逐步修正判断 |

这里要注意一个容易误解的点：代码里仍然有 `env.step()`，因为程序总要模拟环境来返回结果。但 Q-learning 算法本身不把 `env.step()` 当成“可以随便查询所有可能后果的规划工具”。它只是每走一步，接收环境返回的结果，然后更新这一步对应的 Q 值。

这就是 model-free RL 的味道。

## 运行

```bash
python lessons/03_q_learning/q_learning_gridworld.py
```

你会看到类似输出：

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

## 怎么读输出

### config 是什么

| 参数 | 含义 | 先怎么理解 |
| --- | --- | --- |
| `episodes` | 训练多少局 | 每一局从起点开始，到 goal/pit 或达到步数上限结束 |
| `max_steps` | 每局最多走多少步 | 防止 agent 一直绕圈 |
| `alpha` | 学习率 | 新经验对旧 Q 值的影响有多大 |
| `gamma` | 折扣因子 | 越接近 1，越重视未来 reward |
| `epsilon` | 探索概率 | 有多大概率随机试动作 |
| `step_reward` | 每走一步的奖励 | 负数表示走路有成本 |
| `slip_probability` | 动作打滑概率 | 动作可能变成随机方向 |
| `seed` | 随机种子 | 让实验结果更容易复现 |

### 训练日志是什么

```text
episode   500 | avg return last 100:  0.779 | avg steps:   6.0 | success rate:  0.99
```

意思是：

| 字段 | 含义 |
| --- | --- |
| `episode 500` | 已经训练到第 500 局 |
| `avg return last 100` | 最近 100 局的平均累计奖励 |
| `avg steps` | 最近 100 局平均走了多少步 |
| `success rate` | 最近 100 局到达 goal 的比例 |

这里的 `avg return` 不一定单调上升，因为训练时 agent 仍然会用 `epsilon=0.2` 做探索。探索动作可能绕路，也可能掉坑，所以训练日志会有波动。

### Best Q value by state 是什么

`Q(s,a)` 表示：

```text
在状态 s 做动作 a，然后继续按当前学到的好策略走，预计能拿到多少长期回报。
```

每个状态有 4 个 Q 值：

```text
Q(s,U), Q(s,R), Q(s,D), Q(s,L)
```

`Best Q value by state` 打印的是：

```text
max_a Q(s,a)
```

也就是每个格子里“当前最好的动作价值”。

它看起来很像第二课的 `Values`，但来源不同：

```text
第二课 Values：通过已知模型规划出来
第三课 Best Q：通过一条条经验学出来
```

### Greedy policy 是什么

`Greedy policy` 表示训练结束后，每个状态选择 Q 值最大的动作：

```text
best_action = argmax_a Q(s,a)
```

它叫 greedy，是因为最终使用时不再随机探索，而是直接选当前认为最好的动作。

## 查看 Q-learning 的单步更新

先运行一个很短的调试实验：

```bash
python lessons/03_q_learning/q_learning_gridworld.py --episodes 5 --debug-episodes 1 --log-every 0
```

你会看到类似：

```text
Episode 1:
  step  state   action  reward  next    old Q   target  td err  new Q
     1  (2, 0)    L     -0.04 (2, 0)     0.00   -0.04   -0.04   -0.01
     2  (2, 0)    R     -0.04 (2, 1)     0.00   -0.04   -0.04   -0.01
     3  (2, 1)    D     -0.04 (2, 1)     0.00   -0.04   -0.04   -0.01
     4  (2, 1)    U     -0.04 (2, 1)     0.00   -0.04   -0.04   -0.01
```

这一行的意思是：

| 列 | 含义 |
| --- | --- |
| `state` | 当前状态 |
| `action` | agent 这一步选择的动作 |
| `reward` | 环境返回的即时奖励 |
| `next` | 环境返回的下一状态 |
| `old Q` | 更新前的 `Q(state, action)` |
| `target` | 这次经验给出的学习目标 |
| `td err` | target 和 old Q 的差距 |
| `new Q` | 更新后的 Q 值 |

这行输出正好对应代码：

```python
old_q = q[state][action]
next_best = 0.0 if done else max(q[next_state].values())
target = reward + gamma * next_best
td_error = target - old_q
q[state][action] = old_q + alpha * td_error
```

### 为什么 step2 的 old Q 不是 step1 的 new Q？

因为 Q-learning 存的不是“每个 state 一个值”，而是：

```text
每个 state-action 一份值
```

也就是：

```text
Q(state, action)
```

step1 更新的是：

```text
state = (2,0)
action = L
更新的是 Q((2,0), L)
```

所以 step1 的 `new Q = -0.01` 表示：

```text
Q((2,0), L) 从 0.00 变成 -0.01
```

step2 虽然还是在 `(2,0)`，但动作变成了 `R`：

```text
state = (2,0)
action = R
读取的是 Q((2,0), R)
```

`Q((2,0), R)` 在 step2 之前还没有被更新过，所以它的 `old Q` 仍然是 `0.00`。

可以把 Q 表想成这样：

```text
状态 (2,0):
  Q((2,0), U) =  0.00
  Q((2,0), R) =  0.00  <- step2 更新这个
  Q((2,0), D) =  0.00
  Q((2,0), L) = -0.01  <- step1 刚更新的是这个
```

所以：

```text
同一个 state 下，不同 action 有不同 Q 值。
一个动作的 new Q，不会自动变成另一个动作的 old Q。
```

### target 和 step_reward 是什么关系？

`target` 是这次经验给出的“这一步之后，Q 值应该朝哪里靠近”的目标。

公式是：

```text
target = reward + gamma * next_best
```

其中：

```text
reward = 环境这一步实际给的即时奖励
next_best = 下一状态里目前最好的 Q 值
```

在 step1 里：

```text
state = (2,0)
action = L
next = (2,0)
reward = -0.04
```

为什么 `reward = -0.04`？因为 `(2,0)` 向左撞边界，没有到 goal，也没有到 pit，所以环境给普通走路奖励：

```text
step_reward = -0.04
```

训练刚开始时，所有 Q 值都是 `0.00`，所以：

```text
next_best = 0.00
```

于是：

```text
target = reward + gamma * next_best
       = -0.04 + 0.95 * 0.00
       = -0.04
```

所以你看到 step1 的 `target = -0.04`，确实和 `step_reward = -0.04` 有直接关系。但它们不是同一个概念：

| 名称 | 含义 |
| --- | --- |
| `step_reward` | 环境规定的普通走一步即时奖励 |
| `reward` | 这一步实际拿到的即时奖励，可能来自 `step_reward`、goal reward 或 pit reward |
| `target` | 用 `reward + gamma * next_best` 算出的学习目标 |

在训练早期，`next_best` 经常还是 `0.00`，所以 `target` 看起来会等于 `reward`。等 Q 值学起来后，`target` 就会包含下一状态的未来价值，不再只是 `step_reward`。

## 改实验参数：每次只验证一个问题

### 实验 1：改 `episodes`，验证“经验够不够”

```bash
python lessons/03_q_learning/q_learning_gridworld.py --episodes 100 --log-every 50
```

重点看：

```text
Best Q value by state 是否还不稳定？
Greedy policy 是否有局部动作看起来不合理？
```

训练局数少，不代表一定失败，但会更容易出现某些状态经验不足。

### 实验 2：改 `epsilon`，验证“探索是否重要”

```bash
python lessons/03_q_learning/q_learning_gridworld.py --episodes 500 --epsilon 0 --log-every 250
python lessons/03_q_learning/q_learning_gridworld.py --episodes 500 --epsilon 0.5 --log-every 250
```

`epsilon=0` 表示不主动随机探索，只做当前看起来最好的动作。

但注意：本代码在多个动作 Q 值并列时，会随机选一个并列最优动作。所以 `epsilon=0` 不一定完全不探索，尤其在训练早期所有 Q 值都是 `0.0` 时，仍然会因为并列而随机选择。

`epsilon=0.5` 表示一半时间都在随机试动作。它能探索更多，但训练期间的平均回报可能更低，因为随机动作会带来绕路或掉坑。

### 实验 3：改 `alpha`，验证“学习率”

```bash
python lessons/03_q_learning/q_learning_gridworld.py --episodes 1000 --alpha 0.05 --log-every 500
python lessons/03_q_learning/q_learning_gridworld.py --episodes 1000 --alpha 1.0 --log-every 500
```

`alpha` 控制一次新经验对旧 Q 值的影响：

```text
new Q = old Q + alpha * td_error
```

`alpha` 小，学习慢但平滑。  
`alpha` 大，学习快但更容易受单次经验影响。

### 实验 4：改 `slip-probability`，验证“环境随机性”

```bash
python lessons/03_q_learning/q_learning_gridworld.py --slip-probability 0.1 --log-every 1000
```

`slip_probability=0.1` 表示：有 10% 概率动作会变成随机方向。

这会让问题更难，因为同一个 `(state, action)` 不再总是到同一个 `next_state`。你可能会看到：

```text
平均步数变多
平均回报下降
靠近坑的策略更保守
```

## 代码和原理对照

先看源码地图。源码在：

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

整体执行顺序：

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

### 1. Q 表是什么

源码位置：`q_learning_gridworld.py` 第 62-67 行。

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

大概长这样：

```python
q[(2, 0)] = {
    "U": 0.0,
    "R": 0.0,
    "D": 0.0,
    "L": 0.0,
}
```

第二课学的是 `V(s)`：状态本身有多好。  
第三课学的是 `Q(s,a)`：在某个状态做某个动作有多好。

### 2. epsilon-greedy 如何选动作

源码位置：`q_learning_gridworld.py` 第 70-82 行。

```python
if rng.random() < epsilon:
    return rng.choice(ACTIONS)
```

这表示：以 `epsilon` 的概率随机试一个动作。

否则：

```python
values = q[state]
best_value = max(values.values())
best_actions = [action for action, value in values.items() if value == best_value]
return rng.choice(best_actions)
```

这表示：选择当前 Q 值最高的动作。如果多个动作并列最高，就随机选一个。

### 3. 一条经验如何更新 Q 值

源码位置：`q_learning_gridworld.py` 第 111-120 行。

```python
action = choose_action(q, state, epsilon, rng)
next_state, reward, done = env.step(state, action, rng)

old_q = q[state][action]
next_best = 0.0 if done else max(q[next_state].values())
target = reward + gamma * next_best
td_error = target - old_q
q[state][action] = old_q + alpha * td_error
```

逐行解释：

| 代码 | 含义 |
| --- | --- |
| `choose_action(...)` | 根据 epsilon-greedy 选择动作 |
| `env.step(...)` | 环境返回下一状态、奖励、是否结束 |
| `old_q` | 更新前对 `Q(s,a)` 的估计 |
| `next_best` | 下一状态里目前最好的 Q 值 |
| `target` | 这次经验告诉我们的新目标 |
| `td_error` | 新目标和旧估计之间的差距 |
| `q[state][action] = ...` | 把旧估计往新目标方向移动一小步 |

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

### 4. TD error 是什么

```python
td_error = target - old_q
```

可以读成：

```text
这次经验告诉我的目标 - 我原来的估计
```

如果 `td_error > 0`，说明这个动作比原来想的更好，Q 值会上调。  
如果 `td_error < 0`，说明这个动作比原来想的更差，Q 值会下调。

这和第一课的更新形式很像：

```text
new estimate = old estimate + step size * error
```

只不过 Q-learning 的 error 不是 `reward - old estimate`，而是：

```text
reward + gamma * 下一状态最好 Q 值 - old Q
```

### 5. 为什么 Q-learning 是 off-policy

这一节不是为了多记一个术语，而是解释一个关键问题：

```text
训练时 agent 明明会随机探索，
为什么 Q-learning 最后还能学到一个 greedy 的好策略？
```

换句话说，Q-learning 同时做了两件看起来不一样的事：

```text
行为上：为了探索，允许自己乱试动作。
学习上：更新 Q 值时，假设未来会选择当前最好的动作。
```

`off-policy` 这个词就是用来描述这种“不按实际行为方式来定义学习目标”的情况。

训练时，agent 行动用的是 epsilon-greedy：

```text
有时随机探索，有时选择当前最好的动作
```

但学习目标里用的是：

```python
max(q[next_state].values())
```

这表示：更新时假设未来会选择下一状态里最好的动作。

所以：

```text
行为时：可能随机探索
学习时：朝 greedy 最优目标更新
```

行为策略和学习目标不完全一样，这就是 Q-learning 被称为 off-policy 的原因。

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

## 本课过关问题

进入下一课前，你需要能回答：

1. 第二课的 `V(s)` 和第三课的 `Q(s,a)` 有什么区别？
2. 一条经验 `(state, action, reward, next_state)` 在代码里对应哪几行？
3. `target` 为什么是 `reward + gamma * next_best`？
4. `td_error` 表示什么？
5. `epsilon` 为什么影响训练期间的平均回报？
6. `alpha` 太小或太大分别可能带来什么问题？
7. 为什么 Q-learning 不需要提前知道完整环境模型？
8. 为什么 Q-learning 是 off-policy？

## 分享版教程

完整教程见：[docs/articles/lesson_03_q_learning_tutorial.md](../../docs/articles/lesson_03_q_learning_tutorial.md)。
