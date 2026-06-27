# Lesson 02: GridWorld Dynamic Programming

本课目标：从 Bandit 的“只有动作”前进一步，理解 RL 里的 `state`、`value function`、`policy`，以及在知道环境规则时如何用动态规划做 planning。

第一课问的是：

```text
在 5 个动作里，哪个动作总体最好？
```

第二课问的是：

```text
在当前这个格子里，往哪个方向走，长期结果最好？
```

## 这一课解决什么问题

一个机器人在 3x4 网格里移动。它知道地图长什么样，也知道每个动作会把自己带到哪里。

地图如下：

```text
(0,0)  (0,1)  (0,2)  (0,3)=goal +1
(1,0)  WALL   (1,2)  (1,3)=pit  -1
(2,0)  (2,1)  (2,2)  (2,3)
```

动作有 4 个：

| 符号 | 含义 |
| --- | --- |
| `U` | up，向上 |
| `R` | right，向右 |
| `D` | down，向下 |
| `L` | left，向左 |

每走一步默认会拿到 `-0.04` 的奖励，表示“走路有成本”。走到终点 `goal` 得到 `+1`，走到坑 `pit` 得到 `-1`。

这里要先分清两个概念：

```text
我们人类想要的任务目标：让机器人到达终点
RL agent 直接优化的目标：拿到更多累计奖励
```

agent 并不天然理解“终点”这个词。它只看到奖励数字。我们通过 reward design 把“到达终点”翻译成：

```text
到达 goal 给 +1
掉进 pit 给 -1
每多走一步给 -0.04
```

这样 agent 才会倾向于尽快到达 goal，而不是随便乱走。

为什么 RL 这样设计？因为 RL 算法和环境之间的接口通常只有：

```text
agent 选择 action
环境返回 next_state 和 reward
agent 选择让长期 reward 更大的 action
```

也就是说，算法能计算和比较的是数字。它没有办法直接优化“到达终点”这句自然语言，除非我们把这句话变成奖励函数。

在本课里，agent 实际比较的是折扣累计奖励：

```text
return = r1 + gamma * r2 + gamma^2 * r3 + ...
```

`Values` 表里的每个数字，就是从某个格子开始的预计累计奖励。

这一课的核心问题是：

```text
如果我知道整个环境规则，能不能提前算出每个格子该往哪里走？
```

这类问题叫 planning：环境模型已知，不需要先试错收集数据，而是可以直接用规则推算最优策略。

## planning 的反面是什么？

可以先这样理解：

```text
环境模型已知  ->  planning：直接利用规则推算策略
环境模型未知  ->  learning：通过经验学习策略，或者先学习模型再规划
```

这里的“环境模型”指的是：如果 agent 在某个状态 `state` 选择某个动作 `action`，环境会变成哪个 `next_state`，并返回多少 `reward`。

在这一课里，代码的 `GridWorld.step(state, action)` 就是环境模型。因为我们已经写死了地图、墙、终点、陷阱和奖励，所以算法可以在没有真实走路的情况下，直接计算：

```text
如果在这里向上，会到哪里？得到多少 reward？
如果在这里向右，会到哪里？得到多少 reward？
如果在这里向下，会到哪里？得到多少 reward？
如果在这里向左，会到哪里？得到多少 reward？
```

这就是 planning。

相对应地，如果环境模型未知，agent 不知道“做这个动作之后会发生什么”，就不能直接推算。它只能通过数据来学习。常见有三种方式：

1. **model-free RL**：不显式学习环境模型，直接从经验里学习“哪个状态下哪个动作更好”。下一课的 Q-learning 就属于这一类。
2. **model-based RL**：先用数据学习一个环境模型，再用学到的模型做 planning。
3. **offline RL**：不在线试错，而是从已有历史交互数据中学习策略。

所以“环境未知”不一定总是“先试错，再推算最优策略”。更准确地说，是：

```text
模型已知：可以直接规划
模型未知：必须依赖交互数据、历史数据或模拟器数据来学习
```

现实中的典型例子：

| 类型 | 适合场景 | 例子 |
| --- | --- | --- |
| planning / 模型已知 | 规则清楚，动作后果能准确计算 | 已知地图上的路径规划、仓库机器人在固定地图中导航、棋类游戏的规则推演、生产排程中的确定性模拟 |
| model-free RL / 模型未知 | 不知道动作后果，但能不断尝试并得到反馈 | 游戏智能体通过反复玩游戏学习、推荐系统在线试不同内容并观察点击、机器人在仿真环境中学习走路 |
| model-based RL / 先学模型再规划 | 真实规则复杂，但可以用数据拟合一个近似模型 | 自动驾驶用数据学习车辆/交通动态后规划、机械臂学习物体接触动力学后规划动作、工业控制系统学习设备响应模型 |
| offline RL / 历史数据学习 | 真实试错成本高或有风险，只能用已有日志 | 医疗治疗策略分析、金融交易策略研究、自动驾驶从历史驾驶数据中学习 |

本课先讲 planning，是因为它能把 `state`、`action`、`reward`、`value`、`policy` 这些核心概念讲清楚，而且暂时不用处理“数据是怎么采集来的”。下一课开始，我们会去掉“环境模型已知”这个假设，让 agent 只能通过自己走出来的经验学习。

## 运行

```bash
python lessons/02_gridworld_dp/gridworld_dp.py
```

你会看到类似输出：

```text
GridWorld config:
  gamma       : 0.95
  step_reward : -0.04
  goal_reward : 1.0
  pit_reward  : -1.0

Legend:
  U/R/D/L = move up/right/down/left
  +1      = goal terminal state
  -1      = pit terminal state
  WALL    = blocked cell

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

如果你想看 `evaluate_policy()` 内部到底怎么一轮一轮更新 value，可以运行：

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

## 怎么读输出

### config 是什么

```text
gamma       : 0.95
step_reward : -0.04
goal_reward : 1.0
pit_reward  : -1.0
```

含义：

| 参数 | 中文含义 | 直觉 |
| --- | --- | --- |
| `gamma` | 折扣因子 | 越接近 1，越重视未来奖励 |
| `step_reward` | 每走一步的奖励 | 负数表示走路有成本 |
| `goal_reward` | 到达终点奖励 | 正数，表示成功 |
| `pit_reward` | 掉坑奖励 | 负数，表示失败 |

这些参数就是当前案例的奖励定义。默认输出不会逐步打印“这一步拿了多少 reward”，因为这一课不是让机器人真实跑一条轨迹，而是在已知环境模型下做 planning。需要观察单步 reward 时，使用 `--show-evaluation`。

你可以在代码的 `step()` 里看到每一步奖励怎么产生：

```python
reward = self.terminal_rewards.get(next_state, self.step_reward)
```

意思是：

```text
如果 next_state 是终点或坑，就拿对应 terminal reward；
否则就拿 step_reward。
```

### Values 是什么

`Values` 表示每个格子的长期价值，也就是：

```text
从这个格子开始，按照当前算出的策略走，长期大概能拿到多少回报。
```

例如：

```text
  0.82   0.91   1.00   1.00
```

第一行越靠近右边的终点，价值越高。因为离 `goal +1` 更近。

`WALL` 表示墙，不能站上去。`-1.00` 表示坑，是坏终点。

### Policy 是什么

`Policy` 表示每个格子应该采取的动作。

```text
Policy:
  R    R    R    +1
  U  WALL   U    -1
  U    R    U    L
```

例如左下角 `(2,0)` 是 `U`，意思是：

```text
如果机器人在左下角，应该向上走。
```

中下方 `(2,1)` 是 `R`，意思是：

```text
如果机器人在 (2,1)，应该向右走。
```

这张 policy 表就是算法最终算出来的“导航规则”。

### 这和第一课有什么区别

第一课 Bandit 没有状态：

```text
只需要判断哪个动作整体最好。
```

第二课 GridWorld 有状态：

```text
同一个动作在不同格子里的意义不同。
```

例如 `R` 在 `(0,2)` 很好，因为右边就是终点；但 `R` 在 `(1,2)` 很危险，因为右边是坑。

所以从这一课开始，我们不再只学 `Q(a)`，而是开始学习：

```text
V(s) = 某个状态 s 的长期价值
policy(s) = 在状态 s 应该采取什么动作
```

## 改实验参数：每次只验证一个问题

### 实验 1：改 `gamma`，验证“是否重视未来”

`gamma` 是折扣因子，控制未来奖励的重要程度。

```text
gamma = 0.95
```

表示未来奖励很重要。

运行：

```bash
python lessons/02_gridworld_dp/gridworld_dp.py --gamma 0.5
```

这个命令是在问：

```text
如果 agent 更短视，远处格子的价值会不会下降？
```

参考结果里，默认 `gamma=0.95` 时左下角价值是 `0.67`；改成 `gamma=0.5` 后左下角价值变成 `-0.01`。

直觉：

```text
离终点越远，未来奖励越遥远；
如果 gamma 小，远处的未来奖励就被打折得更厉害。
```

这次策略可能不变，但 Values 会明显变小。这也很重要：参数不一定改变最优路线，但会改变算法对每个状态价值的判断。

### 实验 2：改 `step-reward`，验证“走路成本”

`step-reward` 表示每走一步的奖励。默认是 `-0.04`，表示每走一步有轻微成本。

运行：

```bash
python lessons/02_gridworld_dp/gridworld_dp.py --step-reward -0.2
python lessons/02_gridworld_dp/gridworld_dp.py --step-reward -1.0
```

这两个命令是在问：

```text
如果走路成本变高，状态价值会不会下降？
agent 会不会更想尽快结束？
```

参考现象：

- `step_reward=-0.2` 时，Values 整体下降。
- `step_reward=-1.0` 时，远离终点的格子价值会变成明显负数。

这说明：奖励设计会直接影响 agent 对路径的偏好。

### 实验 3：改 `pit-reward`，验证“惩罚大小”

`pit-reward` 表示掉坑的惩罚。默认是 `-1.0`。

运行：

```bash
python lessons/02_gridworld_dp/gridworld_dp.py --pit-reward -5.0
```

这个命令是在问：

```text
如果坑更危险，靠近坑的状态价值会不会更低？
```

在当前确定性地图里，最优策略本来就会避开坑，所以策略可能不变。但坑格子的数值会从 `-1.00` 变成 `-5.00`。

这提醒我们：不要只看 policy 是否变化，也要看 values 如何变化。

### 常见问题 1：为什么我改了参数，Policy 还是一样

这不说明策略永远不受参数影响。

它只说明：在当前这个很小、确定性很强的地图里，默认参数附近的最优路线比较稳定。你改 `gamma`、`step-reward`、`pit-reward` 后，很多时候首先变化的是 `Values`，而不是 `Policy`。

例如：

```text
gamma=0.95 时，左下角价值约为 0.67
gamma=0.5  时，左下角价值约为 -0.01
```

路线可能还是向上、向右走向终点，但 agent 对“这个位置值多少钱”的判断已经变了。

在其他环境里，策略完全可能变化。例如：

- 地图更大，有多条路线；
- 存在绕开坑的长路和靠近坑的短路；
- 走路成本非常高；
- 终点奖励、坑惩罚差距更大；
- 动作有随机性，比如有概率滑向别的方向。

所以要记住：

```text
参数一定会影响价值判断；
是否改变最终策略，取决于环境结构和参数变化幅度。
```

当前代码里，`--step-reward -1.0` 已经会让右下角策略从默认的 `L` 变成 `U`，说明策略并不是完全不受参数影响。

### 常见问题 2：为什么 `--step-reward` 设成正数可能停不下来

不是说 `step-reward` 必须永远是负数。

但在这个任务里，我们是想用奖励设计表达“尽快到达终点”。agent 实际优化的不是“到达终点”这句话，而是累计奖励。

如果每走一步都有正奖励，agent 可能发现：

```text
我一直走路也能拿奖励，那为什么要结束？
```

这会改变问题目标。

例如默认 `gamma=0.95` 时，如果每走一步奖励是 `0.2`，那么“无限走路”的折扣收益大约是：

```text
0.2 / (1 - 0.95) = 4.0
```

而终点奖励只有 `1.0`。这时一直走路比到终点更划算，policy iteration 可能在几种循环策略之间来回切换，无法得到稳定的“去终点”策略。

小的正步进奖励不一定有问题。例如：

```bash
python lessons/02_gridworld_dp/gridworld_dp.py --step-reward 0.01
```

通常仍能收敛，因为：

```text
0.01 / (1 - 0.95) = 0.2
```

还低于终点奖励 `1.0`。

所以更准确的说法是：

```text
如果任务目标是尽快到达终点，step_reward 通常应该是 0 或负数；
如果 step_reward 是正数，需要确保它不会让“永远不结束”比完成任务更划算。
```

当前代码已经对这种情况做了保护：如果策略迭代检测到重复策略，会停止并提示你检查 reward design。

所以这个案例的目标可以更精确地说成：

```text
通过奖励设计，让“到达终点”成为累计奖励最高的行为。
```

如果奖励设计错了，agent 会忠实地优化错误目标。

## 代码和原理对照

先不要从公式开始，也不要从零散代码片段开始。

这一节按源码的执行顺序读。源码在：

```text
lessons/02_gridworld_dp/gridworld_dp.py
```

你可以把整个程序分成三层：

| 层次 | 源码位置 | 作用 | 是否是 RL 核心 |
| --- | --- | --- | --- |
| 入口和输出 | `main()`、`parse_args()`、`print_values()`、`print_policy()` | 读取参数、创建地图、打印结果 | 不是核心，只是让实验能运行、能看懂 |
| 环境模型 | `GridWorld`、`step()` | 定义状态、动作、墙、终点、奖励、动作后果 | 是核心：告诉 agent 世界如何运转 |
| 算法 | `evaluate_policy()`、`improve_policy()`、`policy_iteration()` | 计算价值、改进策略、循环到稳定 | 是核心：真正的 dynamic programming |

所以后面不是随机摘代码，也不是说其他代码没有用，而是先抓住 RL 核心链路：

```text
main()
  -> 创建 GridWorld 环境
  -> policy_iteration()
       -> evaluate_policy()   固定策略，计算 values
       -> improve_policy()    根据 values 改进 policy
       -> 如果 policy 不变，停止
  -> print_values() / print_policy()
```

你现在要重点读的是：

```text
第 24-55 行：GridWorld 和 step()
第 58-97 行：evaluate_policy()
第 100-123 行：improve_policy()
第 126-162 行：policy_iteration()
```

其他代码也有用，但主要是参数解析、打印和错误提示。它们帮助你做实验，不是本节要理解的 RL 原理主体。

### 1. 状态和动作如何表示

源码位置：`gridworld_dp.py` 第 8-17 行。

代码：

```python
Action = str
State = tuple[int, int]

ACTIONS = ["U", "R", "D", "L"]
```

这里：

- `State` 是 `(row, col)`，表示机器人在哪个格子。
- `Action` 是 `"U"`, `"R"`, `"D"`, `"L"` 之一。

例如：

```text
state = (2, 0)
action = "U"
```

表示机器人在左下角，选择向上走。

### 2. 环境规则在哪里

源码位置：`gridworld_dp.py` 第 243-249 行，在 `main()` 里。

代码：

```python
env = GridWorld(
    rows=3,
    cols=4,
    terminal_rewards={(0, 3): args.goal_reward, (1, 3): args.pit_reward},
    walls={(1, 1)},
    step_reward=args.step_reward,
)
```

这定义了：

| 代码 | 含义 |
| --- | --- |
| `rows=3, cols=4` | 3 行 4 列 |
| `(0,3)` | 终点 |
| `(1,3)` | 坑 |
| `(1,1)` | 墙 |
| `step_reward` | 普通移动成本 |

第一课 Bandit 不知道环境真实均值，只能试。  
第二课 GridWorld 知道环境规则，所以可以直接规划。

### 3. `step()` 是环境模型

源码位置：`gridworld_dp.py` 第 43-55 行。

代码：

```python
def step(self, state: State, action: Action) -> tuple[State, float]:
```

它回答一个问题：

```text
如果我在 state 做 action，会到哪里？会拿到多少 reward？
```

核心代码：

```python
dr, dc = DELTAS[action]
next_state = (state[0] + dr, state[1] + dc)
```

如果动作是 `"U"`，`DELTAS["U"] = (-1, 0)`，所以行号减 1，也就是向上。

如果撞墙或出界：

```python
if out_of_bounds or next_state in self.walls:
    next_state = state
```

意思是：撞墙不会移动，仍留在原地。

奖励：

```python
reward = self.terminal_rewards.get(next_state, self.step_reward)
```

如果走到终点或坑，就拿终点奖励；否则拿普通步进奖励。

到这里为止，我们只是在定义“世界如何回应 agent 的动作”。还没有开始算最优策略。

接下来进入算法部分。

### 4. `policy_iteration()` 是总流程

源码位置：`gridworld_dp.py` 第 126-162 行。

先看函数一开始：

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

注意这里跳过了终点和坑：

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

再看外层流程：

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

为什么要先讲这段？因为它是总目录。下面所有代码片段都属于这两个步骤之一。

### 5. `values` 对应 `V(s)`

源码位置：`gridworld_dp.py` 第 67 行，在 `evaluate_policy()` 里面。

代码：

```python
values = {state: 0.0 for state in env.states()}
```

这行代码是在创建一个表。

例如它大概长这样：

```python
values = {
    (0, 0): 0.0,
    (0, 1): 0.0,
    (0, 2): 0.0,
    ...
}
```

`values[state]` 表示“某个格子的长期价值”。

公式里通常写：

```text
V(s)
```

你可以读成：

```text
从状态 s 出发，未来长期大概能拿多少回报。
```

为什么要摘出这一行？因为后面输出里的 `Values` 表，就是这个 `values` 字典算出来的。

它不是普通变量，而是整节课里最重要的数据结构之一：

```text
代码里的 values[state]  <->  公式里的 V(s)  <->  输出里的 Values 表
```

### 6. policy evaluation：固定策略，计算价值

源码位置：`gridworld_dp.py` 第 58-97 行。

先看完整结构：

```python
def evaluate_policy(env, policy, gamma, theta=1e-4):
    values = {state: 0.0 for state in env.states()}

    while True:
        delta = 0.0
        for state in env.states():
            old_value = values[state]
            if env.is_terminal(state):
                values[state] = 0.0
            else:
                action = policy[state]
                next_state, reward = env.step(state, action)
                values[state] = reward + gamma * values[next_state]
            delta = max(delta, abs(old_value - values[state]))
        if delta < theta:
            return values
```

这段函数的目的只有一个：

```text
在 policy 固定不变的情况下，算出每个 state 的长期价值 values。
```

也就是说，它不负责“选择更好的动作”。它只是回答：

```text
如果我一直按照当前 policy 走，每个格子到底值多少钱？
```

其中最关键的是这三行：

代码：

```python
action = policy[state]
next_state, reward = env.step(state, action)
values[state] = reward + gamma * values[next_state]
```

这三行是本课最重要的更新。

翻译成中文：

```text
如果当前策略说在 state 要做 action，
那就看这个 action 会去哪里、拿多少奖励；
当前 state 的价值 = 这一步奖励 + 下一状态价值的折扣。
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

对应简化 Bellman 更新：

```text
V(s) <- r + gamma * V(s')
```

变量对照：

| 公式 | 代码 | 含义 |
| --- | --- | --- |
| `s` | `state` | 当前格子 |
| `a` | `action` | 当前策略选择的动作 |
| `s'` | `next_state` | 下一个格子 |
| `r` | `reward` | 这一步奖励 |
| `gamma` | `gamma` | 未来奖励折扣 |
| `V(s)` | `values[state]` | 当前状态价值 |
| `V(s')` | `values[next_state]` | 下一状态价值 |

完整 RL 书里你会看到更复杂的 Bellman 公式：

```text
V(s) = sum_a pi(a|s) * sum_s' P(s'|s,a) * [r + gamma * V(s')]
```

当前代码没有写这么复杂，是因为本课环境是确定性的：

```text
一个状态 + 一个动作 = 一个确定的下一状态
```

所以代码里只需要：

```text
V(s) <- r + gamma * V(s')
```

#### 为什么这种“自洽”可以评估价值？

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

这就是为什么代码要求：

```python
0 <= gamma < 1
```

如果 `gamma` 等于或超过 1，未来影响不衰减，某些循环任务就可能没有稳定价值。

### 7. 为什么要循环到 `delta < theta`

源码位置：`gridworld_dp.py` 第 69-97 行，仍然在 `evaluate_policy()` 里面。

代码：

```python
delta = max(delta, abs(old_value - values[state]))
if delta < theta:
    return values
```

`delta` 表示这一轮更新里，状态价值变化最大的幅度。

如果所有状态价值都几乎不再变化，说明当前策略的价值已经算稳定了。

`theta` 是停止阈值。默认 `1e-4`，表示变化小到可以忽略。

为什么需要循环？

因为 `values[state]` 依赖 `values[next_state]`。一开始所有价值都是 `0.0`，第一次更新只能得到很粗的估计。多轮更新以后，远处终点或陷阱的影响会一格一格传回来。

可以把它想成：

```text
第 1 轮：离终点近的格子先知道自己比较值钱
第 2 轮：再远一格的格子也受到影响
第 3 轮：影响继续向外传播
...
直到所有格子的价值基本稳定
```

#### 为什么有时会跑 118 轮？

先明确一点：这里不是 agent 真实走了 118 步，也不是试错了 118 次。

`118 sweeps` 的意思是：

```text
在当前固定 policy 下，程序把所有状态的 value 反复重新计算了 118 轮。
```

它本质上是在解一个自洽问题：

```text
每个状态的价值，要和它下一步到达状态的价值互相一致。
```

比如第一轮 policy iteration 的初始策略是“所有格子都向右走”。对于 `(1,0)` 这个格子，右边是墙，所以向右会撞墙，仍然留在 `(1,0)`：

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

注意右边又出现了 `V(1,0)` 自己。这种情况叫“自环”：当前状态的价值依赖当前状态自己。

如果真正解这个方程：

```text
V(1,0) = -0.04 + 0.95 * V(1,0)
```

可以得到：

```text
V(1,0) = -0.8
```

但我们的代码没有直接用代数解方程，而是用迭代逐步逼近：

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

等 value 算稳定以后，才进入下一步 `improve_policy()`，去问：

```text
既然我现在知道每个格子的 value，那有没有更好的动作？
```

#### 为什么不直接走迷宫试错？

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

### 8. policy improvement：根据价值改进策略

源码位置：`gridworld_dp.py` 第 100-123 行。

先看完整结构：

```python
def improve_policy(env, values, policy, gamma):
    stable = True
    new_policy = dict(policy)

    for state in env.states():
        if env.is_terminal(state):
            continue

        old_action = policy[state]
        action_values = {}
        for action in ACTIONS:
            next_state, reward = env.step(state, action)
            action_values[action] = reward + gamma * values[next_state]

        best_action = max(action_values, key=action_values.get)
        new_policy[state] = best_action
        stable = stable and (old_action == best_action)

    return new_policy, stable
```

这个函数的目的和 `evaluate_policy()` 不一样。

`evaluate_policy()` 问：

```text
当前 policy 有多好？
```

`improve_policy()` 问：

```text
既然我已经知道每个 state 的 value，那当前 state 有没有更好的 action？
```

核心代码是：

代码：

```python
for action in ACTIONS:
    next_state, reward = env.step(state, action)
    action_values[action] = reward + gamma * values[next_state]

best_action = max(action_values, key=action_values.get)
new_policy[state] = best_action
```

翻译成中文：

```text
在当前 state，试算 U/R/D/L 四个动作分别会带来多大长期价值；
选价值最高的动作作为新策略。
```

这一步不是实际让机器人走，而是在已知环境模型里“脑内模拟”四个动作。

为什么要摘出这几行？因为 policy improvement 的本质就是：

```text
对同一个 state，把 U/R/D/L 四个动作都算一遍；
谁的 reward + gamma * 下一格价值 最大，就选谁。
```

这里和 `evaluate_policy()` 的区别非常重要：

| 函数 | 是否改变 policy | 看几个动作 |
| --- | --- | --- |
| `evaluate_policy()` | 不改变 | 只看当前 policy 指定的那个动作 |
| `improve_policy()` | 会改变 | 对 U/R/D/L 四个动作都试算 |

这就是为什么第二课叫 policy iteration：

```text
评价当前策略 -> 改进当前策略 -> 再评价 -> 再改进
```

### 9. 回到总流程：policy iteration

代码：

```python
while True:
    values = evaluate_policy(env, policy, gamma)
    policy, stable = improve_policy(env, values, policy, gamma)
    if stable:
        return policy, values, iterations
```

过程是：

```text
先评价当前策略有多好；
再根据评价结果改进策略；
如果策略不再变化，就停止。
```

这就是 policy iteration。

现在你再看公式，逻辑应该是：

```text
1. GridWorld.step() 提供环境模型：s + a -> s' + r
2. evaluate_policy() 用当前 policy 更新 V(s)
3. improve_policy() 用 V(s) 重新选择 action
4. policy_iteration() 让 2 和 3 交替，直到 policy 不变
```

## 本课过关问题

进入第三课前，你需要能回答：

1. `state` 和 `action` 在 GridWorld 里分别是什么？
2. `Values` 表里的数字表示什么？
3. `Policy` 表里的 `U/R/D/L` 表示什么？
4. 为什么 `gamma` 变小后，远处格子的价值会下降？
5. `step_reward` 为什么会影响路径偏好？
6. `V(s) <- r + gamma * V(s')` 对应代码里的哪一行？
7. policy evaluation 和 policy improvement 分别在做什么？

## 分享版教程

完整教程见：[docs/articles/lesson_02_gridworld_dp_tutorial.md](../../docs/articles/lesson_02_gridworld_dp_tutorial.md)。
