# 从 GridWorld 理解强化学习：状态、价值函数和动态规划

第一课我们用多臂老虎机理解了“探索 vs 利用”。Bandit 很适合入门，因为它把问题压缩到最小：每一步只需要从几个动作里选一个。

但真实强化学习问题通常不只是“哪个动作整体最好”。更常见的是：

```text
我现在处在什么状态？
在这个状态下，哪个动作长期更好？
```

这一课用一个小网格世界 GridWorld，学习 RL 里非常核心的三个概念：

- `state`: 状态，agent 当前在哪里；
- `policy`: 策略，在每个状态下该做什么；
- `value function`: 价值函数，从某个状态开始长期能拿多少回报。

## 1. 问题：机器人在网格里找路

地图是一个 3x4 的网格：

```text
(0,0)  (0,1)  (0,2)  (0,3)=goal +1
(1,0)  WALL   (1,2)  (1,3)=pit  -1
(2,0)  (2,1)  (2,2)  (2,3)
```

机器人可以做四个动作：

| 符号 | 含义 |
| --- | --- |
| `U` | 向上 |
| `R` | 向右 |
| `D` | 向下 |
| `L` | 向左 |

奖励规则：

- 每走一步，默认得到 `-0.04`，表示走路有成本；
- 到达右上角终点 `(0,3)`，得到 `+1`；
- 到达坑 `(1,3)`，得到 `-1`；
- 撞墙或出界，就留在原地。

这里要先分清：

```text
人类想要的任务目标：到达终点
agent 实际优化的目标：累计奖励最大
```

agent 不会天然理解“终点”这个词。它只会根据奖励数字做选择。我们把到达终点设置成 `+1`，把掉坑设置成 `-1`，把每走一步设置成 `-0.04`，就是在告诉 agent：

```text
到达终点是好事；
掉坑是坏事；
拖太久也不好。
```

所以“走到终点”不是 agent 直接理解的目标，而是我们通过 reward design 诱导出来的行为。

为什么 RL 要这样做？因为 agent 和环境的交互接口通常很简单：

```text
agent 选择 action
环境返回 next_state 和 reward
agent 学习让长期 reward 更大的 action
```

算法能直接比较的是数字，而不是人类语言里的意图。`goal` 这个词对人有意义，但对算法来说，真正有意义的是：

```text
到达 goal 会得到 +1
```

在这一课里，agent 优化的是折扣累计奖励：

```text
return = r1 + gamma * r2 + gamma^2 * r3 + ...
```

`Values` 表里的数值，就是从每个格子出发，按照当前策略走下去时预计能拿到的累计奖励。

和第一课不同，这一课的 agent 知道地图规则。所以它不需要先在线试错，而是可以直接利用环境规则进行推算。这类问题叫 planning。

planning 对应的是“环境模型已知”。所谓环境模型，就是 agent 在某个状态选择某个动作后，环境会返回什么 `next_state` 和什么 `reward`。

在这一课里，`GridWorld.step(state, action)` 就是环境模型。因为地图、墙、终点、陷阱和奖励都已经写在代码里，所以算法不需要真的让机器人走很多次。它可以直接问：

```text
如果在这里向上，会到哪里？得到多少 reward？
如果在这里向右，会到哪里？得到多少 reward？
如果在这里向下，会到哪里？得到多少 reward？
如果在这里向左，会到哪里？得到多少 reward？
```

然后根据这些规则计算最优策略。

相对应地，也确实存在“环境未知，需要从数据中学习”的问题。但这里要更精确一点：环境未知不一定都是“先试错，再推算最优策略”。常见有三类：

第一类是 **model-free RL**。agent 不显式学习环境模型，而是直接从经验中学习“哪个状态下哪个动作更好”。下一课的 Q-learning 就属于这一类。

第二类是 **model-based RL**。agent 先用数据学习一个近似环境模型，再用这个学到的模型进行 planning。

第三类是 **offline RL**。agent 不在线试错，而是从已有历史交互数据中学习策略。这在真实试错成本高或有风险的场景里很重要。

可以用下面的表对比：

| 类型 | 适合场景 | 现实例子 |
| --- | --- | --- |
| planning / 模型已知 | 规则清楚，动作后果能准确计算 | 已知地图上的路径规划、仓库机器人在固定地图中导航、棋类游戏规则推演、生产排程中的确定性模拟 |
| model-free RL / 模型未知 | 不知道动作后果，但能持续尝试并得到反馈 | 游戏智能体反复玩游戏学习、推荐系统在线试不同内容并观察点击、机器人在仿真环境中学习走路 |
| model-based RL / 先学模型再规划 | 真实规则复杂，但可以用数据拟合近似模型 | 自动驾驶学习车辆/交通动态后规划、机械臂学习接触动力学后规划动作、工业控制系统学习设备响应模型 |
| offline RL / 历史数据学习 | 真实试错成本高或有风险，只能用已有日志 | 医疗治疗策略分析、金融交易策略研究、自动驾驶从历史驾驶数据中学习 |

所以第二课先学 planning，不是因为现实问题都能提前知道环境规则，而是因为 planning 可以把 `state`、`action`、`reward`、`value`、`policy` 这些概念拆得很清楚。等这些概念稳定之后，再进入环境未知的 Q-learning 会更自然。

## 2. 运行实验

在项目根目录运行：

```bash
python lessons/02_gridworld_dp/gridworld_dp.py
```

你会看到：

```text
GridWorld config:
  gamma       : 0.95
  step_reward : -0.04
  goal_reward : 1.0
  pit_reward  : -1.0

Policy iteration converged in 4 iterations.

Values:
  0.82   0.91   1.00   1.00
  0.74   WALL    0.91  -1.00
  0.67   0.74   0.82   0.74

Policy:
  R    R    R    +1
  U  WALL   U    -1
  U    R    U    L
```

如果你想看 `evaluate_policy()` 内部到底怎么更新 value，可以运行：

```bash
python lessons/02_gridworld_dp/gridworld_dp.py --show-evaluation --debug-sweeps 2
```

`--show-evaluation` 表示打印 policy evaluation 的过程。  
`--debug-sweeps 2` 表示每次 evaluation 只展示前 2 轮 sweep，避免输出太长。

你会看到类似片段：

```text
Policy iteration 1: evaluate current policy
  evaluation sweep 1 in policy iteration 1
    state   action  next    reward   old V    new V
    (0, 0)    R    (0, 1)    -0.04    0.00   -0.04
    (0, 1)    R    (0, 2)    -0.04    0.00   -0.04
    (0, 2)    R    (0, 3)     1.00    0.00    1.00
    ...
    max value change delta = 1.000000
```

这段输出的读法：

| 列 | 含义 |
| --- | --- |
| `state` | 当前正在评价哪个格子 |
| `action` | 当前 policy 规定这个格子采取哪个动作 |
| `next` | 执行动作后会到哪个格子 |
| `reward` | 这一步马上得到的奖励 |
| `old V` | 更新前的 `values[state]` |
| `new V` | 用 `reward + gamma * values[next_state]` 算出来的新价值 |
| `delta` | 这一轮 sweep 中最大的 value 变化 |

先看第一轮就够了。它能帮你把这行代码看成具体数字：

```python
values[state] = reward + gamma * values[next_state]
```

## 3. 先看输出，不急着看公式

`Values` 表示每个格子的长期价值。

```text
Values:
  0.82   0.91   1.00   1.00
  0.74   WALL    0.91  -1.00
  0.67   0.74   0.82   0.74
```

你可以把每个数字读成：

```text
如果从这个格子开始，长期大概能拿到多少回报。
```

越靠近右上角终点，价值通常越高。坑是坏终点，所以是负数。

`Policy` 表示每个格子应该往哪里走。

```text
Policy:
  R    R    R    +1
  U  WALL   U    -1
  U    R    U    L
```

例如左下角是 `U`，表示在左下角应该向上走。中下方 `(2,1)` 是 `R`，表示应该向右走。

这张 policy 表就是算法算出来的导航规则。

## 4. 为什么这比 Bandit 更像完整 RL

Bandit 没有状态。它只问：

```text
哪个动作整体最好？
```

GridWorld 有状态。它问：

```text
在当前状态下，哪个动作长期最好？
```

同一个动作在不同状态下意义不同：

- 在 `(0,2)` 做 `R` 很好，因为右边是终点；
- 在 `(1,2)` 做 `R` 很糟，因为右边是坑；
- 在 `(2,3)` 做 `L` 可能比 `U` 更合理，因为可以避开坑。

所以从这一课开始，RL 不再只是估计 `Q(a)`，而是要理解：

```text
V(s): 某个状态的价值
policy(s): 某个状态下该采取的动作
```

## 5. 动态规划在这里做什么

这一课用的是 policy iteration。它由两个步骤反复交替：

```text
policy evaluation   : 固定当前策略，计算每个状态的价值
policy improvement  : 根据价值函数，给每个状态选择更好的动作
```

直到策略不再变化，就停止。

直觉上就是：

1. 先问：如果按当前导航规则走，每个格子值多少钱？
2. 再问：如果我知道每个格子的价值，当前格子是不是有更好的走法？
3. 重复，直到导航规则稳定。

## 6. 按源码执行顺序读代码

先不要从公式开始，也不要从零散代码片段开始。

源码文件是：

```text
lessons/02_gridworld_dp/gridworld_dp.py
```

这个程序可以分成三层：

| 层次 | 作用 | 是否是 RL 核心 |
| --- | --- | --- |
| `main()`、`parse_args()`、`print_values()`、`print_policy()` | 读取参数、创建地图、打印结果 | 不是核心，只是让实验能运行、能看懂 |
| `GridWorld`、`step()` | 定义环境如何回应动作 | 是核心：环境模型 |
| `evaluate_policy()`、`improve_policy()`、`policy_iteration()` | 计算价值、改进策略、循环到稳定 | 是核心：动态规划算法 |

所以后面不是随机摘代码，也不是说其他代码没用。我们只是在先抓住 RL 核心链路：

```text
main()
  -> 创建 GridWorld 环境
  -> policy_iteration()
       -> evaluate_policy()   固定策略，计算 values
       -> improve_policy()    根据 values 改进 policy
       -> 如果 policy 不变，停止
  -> print_values() / print_policy()
```

你可以重点读源码中的这几段：

```text
第 24-55 行：GridWorld 和 step()
第 58-97 行：evaluate_policy()
第 100-123 行：improve_policy()
第 126-162 行：policy_iteration()
```

其他代码主要是参数解析、打印输出和错误提示。它们有用，但不是理解本课 RL 原理的主线。

## 7. 代码里的状态和动作

代码：

```python
Action = str
State = tuple[int, int]

ACTIONS = ["U", "R", "D", "L"]
```

`State` 是 `(row, col)`，例如 `(2,0)` 表示左下角。

`Action` 是 `"U"`、`"R"`、`"D"`、`"L"` 中的一个。

动作如何改变状态，由 `DELTAS` 定义：

```python
DELTAS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
```

例如 `"U"` 让行号减 1，所以是向上。

## 8. 环境模型：`step()`

这节课里，agent 知道环境模型。代码里的环境模型就是：

```python
def step(self, state: State, action: Action) -> tuple[State, float]:
```

它回答：

```text
如果我在 state 做 action，会到哪里？会拿到多少 reward？
```

这就是 model-based planning 里的 model。

核心逻辑：

```python
dr, dc = DELTAS[action]
next_state = (state[0] + dr, state[1] + dc)
```

如果撞墙或出界：

```python
next_state = state
```

奖励：

```python
reward = self.terminal_rewards.get(next_state, self.step_reward)
```

如果下一格是终点或坑，就拿终点奖励；否则拿普通步进奖励。

到这里为止，我们只是在定义“世界如何回应 agent 的动作”。还没有开始算最优策略。

## 9. `policy_iteration()` 是总流程

在源码里，先看函数一开始：

```python
policy = {
    state: "R"
    for state in env.states()
    if not env.is_terminal(state)
}
seen_policies = set()
```

这段代码做了两件事。

第一，创建一个初始策略 `policy`。

`policy` 是一个字典，意思是：

```text
每个非终止状态 -> 当前打算采取的动作
```

这段字典推导式等价于：

```python
policy = {}
for state in env.states():
    if not env.is_terminal(state):
        policy[state] = "R"
```

也就是说，程序一开始先随便给每个普通格子一个动作：全部向右走 `"R"`。

为什么可以随便开始？因为 policy iteration 本来就是：

```text
先给一个不一定好的策略
再评估它
再改进它
反复进行，直到策略稳定
```

所以初始策略不需要一开始就是最优策略。它只是算法的起点。

这里跳过了终点和坑：

```python
if not env.is_terminal(state)
```

因为终点和坑是 terminal state，到了那里 episode 已经结束，不需要再规定“下一步往哪里走”。

第二，创建 `seen_policies`：

```python
seen_policies = set()
```

它是一个集合，用来记录算法以前见过哪些策略。

正常情况下，policy iteration 会收敛到稳定策略。但如果奖励设计有问题，例如 `step_reward` 太大且为正数，agent 可能觉得一直绕圈比到达终点更划算，策略可能反复循环。`seen_policies` 的作用就是检测这种情况，避免程序一直跑下去。

然后再看外层流程：

```python
while True:
    values = evaluate_policy(env, policy, gamma)
    policy, stable = improve_policy(env, values, policy, gamma)
    if stable:
        return policy, values, iterations
```

这段代码说明 policy iteration 只做两件事：

```text
第一步：evaluate_policy   先评价当前 policy 有多好
第二步：improve_policy    再根据评价结果改进 policy
```

为什么先讲这段？因为它是总目录。下面所有代码片段都属于这两个步骤之一。

## 10. 价值函数：`V(s)`

代码里：

```python
values = {state: 0.0 for state in env.states()}
```

`values[state]` 就是状态价值，公式里写成：

```text
V(s)
```

它的含义是：

```text
从状态 s 出发，未来长期能拿到多少回报。
```

这不是当前一步奖励，而是包含未来的长期价值。

为什么要单独讲这一行？因为后面输出里的 `Values` 表，就是这个 `values` 字典算出来的。

它对应关系是：

```text
代码里的 values[state]  <->  公式里的 V(s)  <->  输出里的 Values 表
```

## 11. policy evaluation：固定策略，计算价值

在 `evaluate_policy()` 里：

```python
action = policy[state]
next_state, reward = env.step(state, action)
values[state] = reward + gamma * values[next_state]
```

这三行可以翻译成：

```text
当前状态价值 = 当前这一步奖励 + 下一状态价值的折扣
```

为什么只重点讲这三行？因为它们正好把 RL 的四个核心量连起来：

```text
state -> action -> next_state + reward -> 更新 values[state]
```

其他行也有用：

| 代码 | 作用 |
| --- | --- |
| `while True` | 不断重复更新，直到价值稳定 |
| `for state in env.states()` | 每轮遍历所有格子 |
| `old_value = values[state]` | 记录更新前的价值，用来判断变化大不大 |
| `if env.is_terminal(state)` | 终点和坑不再继续往后走 |
| `delta = ...` | 记录本轮最大变化 |
| `if delta < theta` | 变化很小时停止 |

所以不是其他代码没用，而是这三行最直接对应 Bellman 更新。

公式写成：

```text
V(s) <- r + gamma * V(s')
```

变量对照：

| 公式 | 代码 | 含义 |
| --- | --- | --- |
| `s` | `state` | 当前格子 |
| `a` | `action` | 当前动作 |
| `s'` | `next_state` | 下一格子 |
| `r` | `reward` | 当前一步奖励 |
| `gamma` | `gamma` | 未来奖励折扣 |
| `V(s)` | `values[state]` | 当前状态价值 |
| `V(s')` | `values[next_state]` | 下一状态价值 |

完整 RL 书里常见的 Bellman 公式更复杂：

```text
V(s) = sum_a pi(a|s) * sum_s' P(s'|s,a) * [r + gamma * V(s')]
```

本课代码没有写这个复杂形式，因为环境是确定性的：一个状态加一个动作，只会得到一个确定的下一状态。

所以当前可以先理解简化版：

```text
V(s) <- r + gamma * V(s')
```

为什么这种“自洽”可以评估价值？

关键在于：`V(s)` 的定义本来就是“从状态 `s` 出发，按照当前 policy 走下去，未来能拿到的折扣累计奖励”。

也就是：

```text
V(s) = 从 s 开始的长期回报
```

而长期回报可以拆成两部分：

```text
长期回报 = 眼前这一步 reward + 从下一状态开始的长期回报
```

因为下一状态的长期回报要打折，所以就是：

```text
V(s) = r + gamma * V(s')
```

这不是人为发明出来的技巧，而是价值定义的递归展开。

例如从 `(0,1)` 出发，当前 policy 选择向右：

```text
(0,1) --R--> (0,2)，reward = -0.04
```

那么 `(0,1)` 的价值应该等于：

```text
V(0,1) = -0.04 + 0.95 * V(0,2)
```

如果 `(0,2)` 的价值是 `1.00`，那么：

```text
V(0,1) = -0.04 + 0.95 * 1.00 = 0.91
```

这正好对应输出里的：

```text
Values 第一行: 0.82  0.91  1.00  1.00
```

所以 policy evaluation 其实是在找一组数字，让每个状态都满足这种关系：

```text
当前状态价值 = 当前一步奖励 + 折扣后的下一状态价值
```

如果一组 `values` 满足所有状态的这个关系，那它就不是随便猜的数字，而是当前 policy 下每个状态的长期价值。

为什么可以反复更新逼近这组数字？因为 `gamma < 1`。

`gamma < 1` 的意思是：越远的未来影响越小。每次更新虽然会引用未来价值，但未来价值被乘上 `gamma`，影响会逐步变弱。所以反复更新后，value 会逐渐稳定到同一组答案。

为什么要循环多轮？因为 `values[state]` 依赖 `values[next_state]`。一开始所有价值都是 `0.0`，第一次更新只能得到很粗的估计。多轮更新以后，远处终点或陷阱的影响会一格一格传回来。

有时你会看到：

```text
evaluation converged after 118 sweeps
```

这不是 agent 真实走了 118 步，也不是试错了 118 次。它的意思是：在当前固定 policy 下，程序把所有状态的 value 反复重新计算了 118 轮。

本质上，`evaluate_policy()` 在解一个自洽问题：

```text
每个状态的价值，要和它下一步到达状态的价值互相一致。
```

比如第一轮 policy iteration 的初始策略是“所有格子都向右走”。对于 `(1,0)`，右边是墙，所以向右会撞墙，仍然留在 `(1,0)`：

```text
state = (1,0)
action = R
next_state = (1,0)
reward = -0.04
```

于是更新公式变成：

```text
V(1,0) <- -0.04 + 0.95 * V(1,0)
```

右边又出现了 `V(1,0)` 自己。这种情况叫“自环”：当前状态的价值依赖当前状态自己。

如果直接解方程：

```text
V(1,0) = -0.04 + 0.95 * V(1,0)
```

可以得到：

```text
V(1,0) = -0.8
```

但代码没有直接用代数解方程，而是用迭代逐步逼近：

```text
第 1 轮：V = -0.04
第 2 轮：V = -0.04 + 0.95 * (-0.04) = -0.078
第 3 轮：V = -0.04 + 0.95 * (-0.078) = -0.1141
...
最后慢慢接近 -0.8
```

因为 `gamma = 0.95` 很接近 1，每一轮还会保留很多未来价值，所以收敛会比较慢。`theta = 1e-4` 又要求变化小到 `0.0001` 以下才停止，所以第一轮 policy evaluation 可能需要很多 sweep。

所以，`evaluate_policy()` 的本质不是“找更好的策略”，而是：

```text
在当前 policy 不变的前提下，
反复更新每个状态的 value，
直到这些 value 互相一致、基本不再变化。
```

为什么不直接走迷宫试错？

因为这一课的前提是：环境模型已知。

代码里已经有：

```python
next_state, reward = env.step(state, action)
```

这意味着程序可以直接问环境模型：

```text
如果我在 state 做 action，会到哪里？会得到多少 reward？
```

既然规则已经知道，就没有必要通过真实走很多次来估计结果。

如果用“走迷宫试错”的方式评估当前 policy，大概会变成：

```text
从某个 state 出发
按照当前 policy 一直走
记录一路拿到的 reward
重复很多次
用平均结果估计这个 state 的 value
```

这种方法当然也可以，它接近后面会学的 Monte Carlo / model-free learning。但在本课这个场景里，它有几个问题：

| 方法 | 需要环境模型吗 | 主要代价 | 适合场景 |
| --- | --- | --- | --- |
| 反复计算 value | 需要 | 计算量 | 环境规则已知，可以直接推算 |
| 走迷宫试错 | 不需要 | 采样次数、随机误差、真实试错成本 | 环境规则未知，只能通过经验学习 |

所以本课选择反复计算 value，不是因为 RL 永远都这样做，而是因为：

```text
已知模型时，用模型直接算，比靠试错估计更干净、更稳定、更省样本。
```

等到第三课 Q-learning，我们会去掉这个假设。那时 agent 不再提前知道完整环境模型，只能拿到自己每一步实际经历的：

```text
state, action, reward, next_state
```

那就必须通过“走出来的数据”学习，而不能像本课这样直接规划。

## 12. policy improvement

有了状态价值后，下一步是改进策略。

代码：

```python
for action in ACTIONS:
    next_state, reward = env.step(state, action)
    action_values[action] = reward + gamma * values[next_state]

best_action = max(action_values, key=action_values.get)
new_policy[state] = best_action
```

含义：

```text
在当前状态，分别试算 U/R/D/L 四个动作的长期价值；
选择价值最高的动作作为新策略。
```

注意，这里不是实际让机器人走四次，而是在已知环境模型里计算四种可能结果。

这里和 `evaluate_policy()` 的区别非常重要：

| 函数 | 是否改变 policy | 看几个动作 |
| --- | --- | --- |
| `evaluate_policy()` | 不改变 | 只看当前 policy 指定的那个动作 |
| `improve_policy()` | 会改变 | 对 U/R/D/L 四个动作都试算 |

所以完整逻辑是：

```text
1. GridWorld.step() 提供环境模型：s + a -> s' + r
2. evaluate_policy() 用当前 policy 更新 V(s)
3. improve_policy() 用 V(s) 重新选择 action
4. policy_iteration() 让 2 和 3 交替，直到 policy 不变
```

## 13. 改 gamma：短视和长期主义

运行：

```bash
python lessons/02_gridworld_dp/gridworld_dp.py --gamma 0.5
```

默认 `gamma=0.95` 时，左下角价值是 `0.67`。改成 `gamma=0.5` 后，左下角价值变成 `-0.01`。

原因是：左下角离终点远，终点奖励属于较远的未来。如果 `gamma` 小，未来奖励会被严重打折。

所以 `gamma` 可以理解为：

```text
我有多重视未来？
```

## 14. 改 step_reward：走路成本

运行：

```bash
python lessons/02_gridworld_dp/gridworld_dp.py --step-reward -0.2
python lessons/02_gridworld_dp/gridworld_dp.py --step-reward -1.0
```

`step_reward` 越负，走路成本越高。你会看到远离终点的格子价值明显下降。

当 `step_reward=-1.0` 时，有些状态甚至会倾向于更快结束，而不是绕远路。这不是算法坏了，而是 reward design 改变了问题目标：

```text
如果每多走一步都非常痛苦，快速结束可能比长距离绕路更划算。
```

这说明 reward design 很重要。你给什么奖励，agent 就会围绕什么目标优化。

## 15. 改 pit_reward：惩罚大小

运行：

```bash
python lessons/02_gridworld_dp/gridworld_dp.py --pit-reward -5.0
```

在当前地图里，最优策略本来就会避开坑，所以 policy 可能不变。但坑本身的价值会从 `-1.00` 变成 `-5.00`。

这说明分析 RL 结果时不能只看策略是否变化，也要看价值函数如何变化。

## 16. 两个容易误解的实验现象

### 改参数后策略为什么经常一样

这不说明策略永远不受参数影响。

在当前这个 3x4 确定性小地图里，最优路线比较稳定。很多参数变化会先改变 `Values`，但不一定改变 `Policy`。

例如 `gamma=0.95` 时，左下角价值约为 `0.67`；`gamma=0.5` 时，左下角价值约为 `-0.01`。路线可能仍然是朝终点走，但这个状态的长期价值已经明显下降。

在更复杂的地图里，策略可能会变化。例如存在长路和短路、危险区域、随机滑动、多个终点时，`gamma`、步进成本和惩罚大小都可能改变最终路线。

所以正确理解是：

```text
参数一定会影响价值判断；
是否改变策略，取决于环境结构和参数变化幅度。
```

### step_reward 能不能是正数

不是绝对不能。

但在“尽快到达终点”的任务里，正的 step reward 很危险。因为 agent 实际优化的是累计奖励，不是“到达终点”这句话。正步进奖励可能让 agent 觉得一直走路比结束更好。

例如默认 `gamma=0.95`，如果 `step_reward=0.2`，那么无限走路的折扣收益大约是：

```text
0.2 / (1 - 0.95) = 4.0
```

而终点奖励只有 `1.0`。这时一直走路比到终点更划算，任务目标就被 reward design 改坏了。

小的正数不一定会出问题。例如 `step_reward=0.01` 时，无限走路的折扣收益大约是 `0.2`，仍然低于终点奖励。

所以规则不是“必须为负数”，而是：

```text
如果希望 agent 完成任务，reward 设计不能让不完成任务更划算。
```

当前代码会检测重复策略。如果正步进奖励导致 policy iteration 在循环策略之间来回切换，程序会停止并提示你检查 reward design。

所以这个案例的目标更准确地说是：

```text
通过奖励设计，让到达终点成为累计奖励最高的行为。
```

如果 reward design 让“永远走路”更高分，agent 就会优化这个错误目标。

## 17. 本课你应该带走什么

第一，GridWorld 引入了状态。RL 不再只是“哪个动作平均最好”，而是“在当前状态下哪个动作长期最好”。

第二，`V(s)` 是状态价值，不是一步奖励。它包含未来。

第三，如果环境模型已知，可以用 planning 方法直接算策略。policy iteration 就是一个典型方法。

第四，Bellman 更新不要先背公式，先看代码：

```python
values[state] = reward + gamma * values[next_state]
```

这就是：

```text
当前价值 = 当前奖励 + 折扣后的下一状态价值
```

第五，reward 和 gamma 的设置会改变价值判断，有时也会改变策略。

## 18. 进入下一课前的问题

你应该能回答：

1. Bandit 为什么没有状态，而 GridWorld 有状态？
2. `Values` 表里的数字是什么意思？
3. `Policy` 表里的箭头是什么意思？
4. `gamma` 小了为什么远处状态价值下降？
5. policy evaluation 和 policy improvement 分别做什么？
6. `values[state] = reward + gamma * values[next_state]` 为什么是 Bellman 更新？

下一课会去掉一个关键假设：agent 不再提前知道环境模型。那时我们会进入 Q-learning。
