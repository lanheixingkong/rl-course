# RL 实战课程学习计划

这个计划不是固定课表，而是当前阶段的学习操作指南。后续会根据学习过程持续调整。

## 总体学习方法

每一课都要同时看文档和代码。它们分工不同：

- 根目录 `README.md`: 看课程整体目标、路线、当前该学哪一课。
- 每课目录下的 `README.md`: 看这一课要解决的问题、运行命令、观察重点、练习。
- `docs/articles/...`: 看完整中文教程，适合系统理解和分享。
- `.py` 代码: 用来运行实验、改参数、读算法实现。
- `docs/study_plan.md`: 看当前阶段的具体学习步骤和过关标准。

每一课按这个顺序推进：

1. 先读根目录 `README.md` 的课程路线，确认当前学习位置。
2. 读本课 `README.md`，只读到能知道“这课解决什么问题、运行什么命令、看什么指标”即可。
3. 运行代码，不先看公式。
4. 观察输出，描述发生了什么。
5. 改 2-3 个关键参数，确认现象是否稳定。
6. 回到代码，找到现象对应的实现位置。
7. 阅读分享版教程，补齐原理、术语和适用边界。
8. 用自己的话回答过关问题。
9. 做一个小改造，把方法迁移到相近问题。

不要把重点放在记公式。重点是建立这条链路：

```text
实际问题 -> 建模方式 -> 可运行代码 -> 实验现象 -> 原理解释 -> 迁移判断
```

## 当前学习位置

当前正在学习：

```text
Lesson 03: Q-learning GridWorld
```

材料：

- `README.md`
- `lessons/03_q_learning/README.md`
- `lessons/03_q_learning/q_learning_gridworld.py`
- `docs/articles/lesson_03_q_learning_tutorial.md`
- `docs/study_plan.md`

## Lesson 03 学习步骤

### Step 0: 先读这些文件

按顺序读：

1. `lessons/03_q_learning/README.md`
   - 先读到“怎么读输出”即可。
   - 目的：知道 Q-learning 输出里的 `avg return`、`success rate`、`Best Q value`、`Greedy policy` 是什么。
2. `docs/study_plan.md`
   - 读当前 Lesson 03 部分。
   - 目的：知道这一课具体要跑哪些实验。

这一步不要先背 Q-learning 公式。先运行代码，看 agent 如何从经验中学习。

### Step 1: 运行默认实验

```bash
python lessons/03_q_learning/q_learning_gridworld.py
```

运行后先回到 `lessons/03_q_learning/README.md` 的“运行”和“怎么读输出”部分，对照看。

先确认你理解：

```text
episode = 从起点开始的一局
avg return = 最近若干局的平均累计奖励
success rate = 到达 goal 的比例
Best Q value = 每个状态下当前最好的动作价值
Greedy policy = 训练后选择 Q 值最大动作得到的策略
```

### Step 2: 看单步 Q 更新

```bash
python lessons/03_q_learning/q_learning_gridworld.py --episodes 5 --debug-episodes 1 --log-every 0
```

重点看这一行：

```text
state, action, reward, next, old Q, target, td err, new Q
```

先不要急着看公式。先用中文解释：

```text
这一步发生了什么？
old Q 为什么会变成 new Q？
target 和 td err 分别表示什么？
```

### Step 3: 改 episodes

目的：验证“经验数量是否足够”。

```bash
python lessons/03_q_learning/q_learning_gridworld.py --episodes 100 --log-every 50
```

重点看：

```text
Best Q value 是否还有些状态不稳定？
Greedy policy 是否有局部动作看起来不合理？
```

### Step 4: 改 epsilon

目的：验证“探索”的影响。

```bash
python lessons/03_q_learning/q_learning_gridworld.py --episodes 500 --epsilon 0 --log-every 250
python lessons/03_q_learning/q_learning_gridworld.py --episodes 500 --epsilon 0.5 --log-every 250
```

注意：`epsilon=0` 不一定完全没有随机性。因为初始 Q 值相同，代码会在并列最优动作中随机选一个。

### Step 5: 改 alpha

目的：验证“学习率”的影响。

```bash
python lessons/03_q_learning/q_learning_gridworld.py --episodes 1000 --alpha 0.05 --log-every 500
python lessons/03_q_learning/q_learning_gridworld.py --episodes 1000 --alpha 1.0 --log-every 500
```

重点看：

```text
学习是否变慢？
训练日志是否更波动？
最终 greedy policy 是否稳定？
```

### Step 6: 改 slip_probability

目的：验证“环境随机性”。

```bash
python lessons/03_q_learning/q_learning_gridworld.py --slip-probability 0.1 --log-every 1000
```

重点看：

```text
avg steps 是否变多？
avg return 是否下降？
靠近坑的策略是否更保守？
```

### Step 7: 回到代码

打开 `lessons/03_q_learning/q_learning_gridworld.py`，按顺序找这些部分：

- `GridWorld.step`
- `make_q_table`
- `choose_action`
- `train_q_learning`

重点读这几行：

```python
old_q = q[state][action]
next_best = 0.0 if done else max(q[next_state].values())
target = reward + gamma * next_best
td_error = target - old_q
q[state][action] = old_q + alpha * td_error
```

学习顺序仍然是：先理解代码变量，再看公式。

### Step 8: 阅读完整教程

读：

```text
docs/articles/lesson_03_q_learning_tutorial.md
```

重点读：

- 第 1-5 节：理解 Q-learning 解决的问题和输出。
- 第 6-10 节：理解代码执行顺序、Q 表、TD error。
- 第 11-13 节：理解 model-free、off-policy 和参数实验。

## Lesson 03 过关标准

进入下一课前，你需要能回答：

1. `V(s)` 和 `Q(s,a)` 的区别是什么？
2. 为什么 Q-learning 可以从单条经验更新？
3. `target = reward + gamma * next_best` 表达了什么？
4. `td_error` 为正和为负分别意味着什么？
5. `epsilon` 太小和太大分别有什么问题？
6. `alpha` 太小和太大分别有什么问题？
7. Q-learning 为什么不需要提前知道完整环境模型？
8. Q-learning 为什么叫 off-policy？

## Lesson 02 学习步骤

### Step 0: 先读这些文件

按顺序读：

1. `lessons/02_gridworld_dp/README.md`
   - 先读到“怎么读输出”即可。
   - 目的：知道 GridWorld 里的状态、动作、Values、Policy 分别是什么。
2. `docs/study_plan.md`
   - 读当前 Lesson 02 部分。
   - 目的：知道这一课具体要跑哪些实验。

这一步不要先读完整分享版教程。先跑程序，看输出，再回来补原理。

### Step 1: 运行默认实验

```bash
python lessons/02_gridworld_dp/gridworld_dp.py
```

运行后先回到 `lessons/02_gridworld_dp/README.md` 的“运行”和“怎么读输出”部分，对照看。

先确认你理解：

```text
Values = 每个格子的长期价值
Policy = 每个格子应该采取的动作
U/R/D/L = 上/右/下/左
```

先回答：

```text
为什么越靠近终点的格子价值通常越高？
为什么同一个动作在不同格子里好坏不同？
```

### Step 2: 改 gamma

目的：验证“未来奖励的重要程度”。

```bash
python lessons/02_gridworld_dp/gridworld_dp.py --gamma 0.5
```

重点看：

```text
Values 表里远离终点的格子价值是否明显下降？
Policy 是否一定会变化？
```

### Step 3: 改走路成本

目的：验证 reward design 会改变状态价值。

```bash
python lessons/02_gridworld_dp/gridworld_dp.py --step-reward -0.2
python lessons/02_gridworld_dp/gridworld_dp.py --step-reward -1.0
```

重点看：

```text
Values 是否整体下降？
远离终点的格子是否更容易变成负数？
```

不要把 `step-reward` 随便改成较大的正数。正步进奖励可能让 agent 觉得“永远走路拿奖励”比到达终点更划算。如果要测试正数，可以先试很小的值：

```bash
python lessons/02_gridworld_dp/gridworld_dp.py --step-reward 0.01
```

然后再理解为什么 `--step-reward 0.2` 会触发 reward design 问题。

### Step 4: 改坑的惩罚

目的：验证惩罚大小如何影响价值。

```bash
python lessons/02_gridworld_dp/gridworld_dp.py --pit-reward -5.0
```

重点看：

```text
坑格子的价值是否变化？
Policy 是否一定变化？
为什么当前确定性地图里策略可能不变？
```

关键提醒：

```text
参数变化一定会影响价值判断；
但不一定每次都改变最终策略。
```

### Step 5: 回到代码

打开 `lessons/02_gridworld_dp/gridworld_dp.py`，按顺序找这些部分：

- `ACTIONS` 和 `DELTAS`
- `GridWorld.step`
- `evaluate_policy`
- `improve_policy`
- `policy_iteration`

先读这行 Bellman 更新对应的代码：

```python
values[state] = reward + gamma * values[next_state]
```

再读这段策略改进：

```python
for action in ACTIONS:
    next_state, reward = env.step(state, action)
    action_values[action] = reward + gamma * values[next_state]

best_action = max(action_values, key=action_values.get)
```

学习顺序仍然是：先理解代码变量，再看公式。

### Step 6: 阅读完整教程

读：

```text
docs/articles/lesson_02_gridworld_dp_tutorial.md
```

重点读：

- 第 1-4 节：理解 GridWorld、Values、Policy。
- 第 5-10 节：理解 policy iteration、Bellman 更新、代码对照。
- 第 11-13 节：理解参数实验。

### Step 7: 写学习笔记

建议写 5 句话：

1. GridWorld 相比 Bandit 多了什么。
2. `Values` 表里的数字表示什么。
3. `Policy` 表里的动作表示什么。
4. `gamma` 和 `step_reward` 分别影响什么。
5. policy evaluation 和 policy improvement 分别做什么。

## Lesson 02 过关标准

进入第三课前，你需要能回答：

1. `state` 和 `action` 在 GridWorld 里分别是什么？
2. 为什么 `V(s)` 不是当前一步奖励，而是长期价值？
3. `values[state] = reward + gamma * values[next_state]` 在表达什么？
4. `gamma` 变小为什么会让远处状态价值下降？
5. `step_reward` 为什么是 reward design 的一部分？
6. policy evaluation 和 policy improvement 的区别是什么？
7. 为什么本课叫 planning？

## Lesson 01 学习步骤

### Step 0: 先读这些文件

按顺序读：

1. `README.md`
   - 只看“学习方式”“课程路线”“快速开始”。
   - 目的：知道整个课程不是按理论章节推进，而是按可运行案例推进。
2. `lessons/01_bandit/README.md`
   - 全部读完。
   - 目的：知道 Bandit 解决什么问题、要运行什么命令、要观察哪些指标。
3. `docs/study_plan.md`
   - 读当前 Lesson 01 部分。
   - 目的：知道今天具体该做什么，以及什么时候算过关。

这一步不要先读完整分享版教程。先保留一点问题感，等跑完实验后再读。

### Step 1: 运行默认实验

在项目根目录运行：

```bash
python lessons/01_bandit/bandit.py
```

运行后先回到 `lessons/01_bandit/README.md` 的“运行”部分，阅读示例输出和逐行解释。不要直接跳到改参数。

先确认你理解：

```text
5 个动作 = 每一步可选的 5 个选项
2000 steps = 在这 5 个选项里重复选择 2000 次
```

你需要观察三个指标：

- `avg total reward`: 平均总收益，越高越好。
- `avg regret`: 相比理论最优损失了多少，越低越好。
- `avg best action rate`: 选择真实最优动作的比例，越高越好。

先不要急着看公式。先回答：

```text
greedy、epsilon-greedy、UCB 谁表现最好？
greedy 为什么看起来聪明但平均表现差？
```

### Step 2: 改探索率

目的：验证“探索太少或太多都不好”。

`epsilon` 表示 `epsilon_greedy` 随机探索的比例。`epsilon=0.1` 表示 10% 的时候随机试试，90% 的时候选当前看起来最好的动作。

运行：

```bash
python lessons/01_bandit/bandit.py --epsilon 0.01
python lessons/01_bandit/bandit.py --epsilon 0.3
```

重点只看 `epsilon_greedy` 的：

```text
avg total reward
avg regret
avg best action rate
```

要观察：

```text
epsilon 太小时，是否探索不足？
epsilon 太大时，是否探索过多？
为什么 epsilon=0.1 在这个实验里更平衡？
```

### Step 3: 改决策步数

目的：验证“任务越长，早期探索越值得”。

`steps` 表示一轮实验里重复选择多少次。动作仍然只有 5 个，只是在这 5 个动作里反复选。

运行：

```bash
python lessons/01_bandit/bandit.py --steps 200
python lessons/01_bandit/bandit.py --steps 10000
```

重点比较三种策略的：

```text
avg regret
avg best action rate
```

要观察：

```text
为什么步数越长，早期探索越值得？
短期收益和长期收益的权衡在哪里？
```

### Step 4: 改奖励噪声

目的：验证“反馈越随机，越不能相信早期结果”。

`reward-std` 表示奖励波动大小。值越大，同一个动作每次给出的奖励越不稳定。

运行：

```bash
python lessons/01_bandit/bandit.py --reward-std 2.0
```

重点看：

```text
avg regret
last-run estimates
last-run counts
```

要观察：

```text
反馈越 noisy，为什么越不能相信早期结果？
```

### Step 5: 改 UCB 探索强度

目的：验证“UCB 的探索强度也不是越大越好”。

`ucb-c` 只影响 `ucb` 策略。它控制 UCB 公式里探索奖励的强弱。

运行：

```bash
python lessons/01_bandit/bandit.py --ucb-c 0.5
python lessons/01_bandit/bandit.py --ucb-c 4.0
```

重点只看 `ucb` 的：

```text
avg total reward
avg regret
avg best action rate
last-run counts
```

要观察：

```text
ucb-c 太小时，是否更偏向利用？
ucb-c 太大时，是否探索过多？
为什么探索强度也需要调节？
```

### Step 6: 回到代码

打开 `lessons/01_bandit/bandit.py`，找到这些函数：

- `choose_greedy`
- `choose_epsilon_greedy`
- `choose_ucb`
- `run_bandit`

先读三个策略函数，理解它们分别如何选动作：

```text
choose_greedy          -> 只选当前估计最好的动作
choose_epsilon_greedy  -> 少数时候随机探索，大多数时候 greedy
choose_ucb             -> 同时看估计值和尝试次数，优先探索不确定动作
```

再重点读这一行：

```python
estimates[action] += (reward - estimates[action]) / counts[action]
```

它表示：

```text
新估计 = 旧估计 + 本次奖励和旧估计的差距 / 该动作被尝试次数
```

然后回到 `lessons/01_bandit/README.md` 的“代码和原理对照”部分，对照阅读：

```text
true_means       -> q*(a)，动作真实平均奖励
estimates        -> Q(a)，agent 当前估计
counts           -> N(a)，动作被尝试次数
reward           -> r，这一步拿到的奖励
regret           -> 理论最优收益 - 实际收益
```

学习顺序是：先理解代码变量，再看公式。不要反过来硬背公式。

这行更新代码是后面 TD learning 和 Q-learning 的雏形。

### Step 7: 阅读完整教程

读：

```text
docs/articles/lesson_01_bandit_tutorial.md
```

重点读这些部分：

- 第 1-4 节：理解问题、实验结果、三种策略。
- 第 5-6 节：理解 `Q(a)` 和 regret。先用 README 的代码对照理解，再读文章里的原理解释。
- 第 11-14 节：理解为什么 Bandit 属于 RL、和传统 ML/神经网络的关系、什么时候不该用 RL。

读的时候不要追求记住每个公式。你要把这些问题弄清楚：

```text
这个问题为什么不是普通监督学习？
为什么“数据如何产生”是关键？
为什么探索本身有成本，但仍然值得？
```

### Step 8: 写学习笔记

建议在自己的笔记里写 5 句话：

1. Bandit 问题是什么。
2. greedy 的问题是什么。
3. epsilon-greedy 如何解决一部分问题。
4. UCB 比随机探索多考虑了什么。
5. 什么情况下应该优先用传统 ML，什么情况下考虑 Bandit/RL。

## Lesson 01 过关标准

你可以进入 Lesson 02 的标准不是“会背 UCB 公式”，而是能回答这些问题：

1. 为什么 Bandit 是 RL 入门问题，而不只是一个普通预测问题？
2. 为什么有完整、无偏、覆盖充分的数据时，应该优先考虑传统 ML？
3. 为什么只有被选择动作有反馈时，需要考虑探索？
4. greedy 为什么可能长期表现差？
5. epsilon 太小和太大分别有什么问题？
6. UCB 比随机探索多考虑了什么？
7. regret 衡量的是什么？

如果这些问题回答得比较顺，可以进入 GridWorld。

## Lesson 01 小改造

建议做一个小改造，不需要复杂：

```text
把 true_means 改成更接近的数值，例如 [1.0, 1.1, 1.2, 1.3, 1.4]
```

然后重新运行默认实验。

观察：

```text
动作之间差距变小时，哪个策略更受影响？
为什么区分接近的动作需要更多探索？
```

## Lesson 02 预告

Bandit 没有状态。下一课 GridWorld 会引入：

- `state`: agent 当前在哪里；
- `action`: agent 可以往哪个方向走；
- `reward`: 到达终点、掉坑、每走一步的成本；
- `transition`: 动作如何改变状态；
- `value function`: 从某个状态开始，长期能拿到多少回报。

从 Lesson 01 到 Lesson 02 的关键变化是：

```text
Bandit: 哪个动作总体最好？
GridWorld: 在当前状态下，哪个动作带来的长期结果最好？
```
