# 实战型 RL 教程设计

## 总目标

完成这套教程后，你应该能做到三件事：

1. 读懂别人使用 RL 方法时在优化什么、状态动作奖励如何定义、为什么选这个算法。
2. 面对一个新问题，可以判断它是否需要 RL，而不是把所有问题都套成 RL。
3. 能搭建一个最小可行 baseline，再根据问题规模升级到 DQN、Policy Gradient、Actor-Critic 或 PPO。

## 判断一个问题是否适合 RL

RL 适合的问题通常有这些特征：

- 决策会影响未来，而不只是当前一步。
- 行动后得到的反馈可能延迟出现。
- 环境有不确定性，或者没有明确的最优规则。
- 可以反复试验，或者至少可以在模拟器里试验。
- 目标可以用奖励函数描述。

不适合直接用 RL 的情况：

- 每个样本都已有明确标签：先考虑监督学习。
- 只需要一次性排序、分类、回归：先考虑传统 ML。
- 试错成本极高且没有模拟器：先考虑优化、规划、离线评估或安全约束方法。
- 奖励无法定义，或者会被策略轻易钻空子：先重新定义问题。

## 一个 RL 问题的最小建模模板

拿到新问题时，先写下面五项：

```text
State: agent 做决策时能观察到什么？
Action: agent 可以选择什么？
Reward: 一步之后如何评价这个动作？
Transition: 动作如何改变世界？是否已知？
Episode end: 什么时候一轮结束？
```

再补充三个工程问题：

```text
Baseline: 不用 RL 的简单规则是什么？
Metric: 训练之外，最终用什么指标评价？
Risk: 错误动作的成本是什么，如何限制？
```

## 课程结构

### 1. Bandit：最小 RL 问题

案例：在多个广告/推荐/药物剂量/老虎机中选择一个，每次只看到被选动作的收益。

核心问题：

- 为什么不能总选当前看起来最好的动作？
- 随机探索的成本是什么？
- epsilon-greedy 的 epsilon 如何影响长期收益？

学完能看懂：

- exploration / exploitation
- action value estimate
- regret
- online learning

### 2. GridWorld 动态规划：知道环境模型时如何规划

案例：小机器人在网格中移动，避开坑，到达终点。

核心问题：

- 如果知道每个动作会把你带到哪里，如何计算每个状态的价值？
- policy evaluation 和 policy improvement 为什么能交替变好？
- Bellman equation 到底在表达什么？

学完能看懂：

- MDP
- value function `V(s)`
- policy `pi(a|s)`
- discount factor `gamma`
- Bellman backup

### 3. Q-learning：不知道环境模型时如何边试边学

案例：同样的 GridWorld，但 agent 不再提前知道转移规则，只能靠与环境交互学习。

核心问题：

- 如何用一次经验 `(s, a, r, s')` 更新长期价值？
- TD error 是什么？
- 为什么 Q-learning 是 off-policy？

学完能看懂：

- action-value function `Q(s, a)`
- temporal difference learning
- learning rate `alpha`
- epsilon schedule
- off-policy control

### 4. 函数逼近：当状态太多时怎么办

案例：从小网格变成连续状态或高维观测。

核心问题：

- 为什么表格 Q 不再可行？
- 神经网络在 RL 里近似什么？
- 为什么 RL 训练比监督学习更不稳定？

### 5. DQN：把 Q-learning 接到神经网络

案例：CartPole 或简化游戏环境。

核心问题：

- replay buffer 为什么重要？
- target network 解决什么问题？
- epsilon decay 如何影响训练？

### 6. Policy Gradient：直接学习动作概率

案例：连续控制或策略随机性很重要的问题。

核心问题：

- 为什么有时不学 Q，而是直接学 policy？
- return-to-go 和 baseline 如何降低方差？

### 7. Actor-Critic / PPO：实用深度 RL 主线

案例：用 PPO 解决一个 Gymnasium 任务，再迁移到自定义环境。

核心问题：

- actor 和 critic 分别学什么？
- advantage 的含义是什么？
- PPO 的 clip 为什么能稳定训练？

### 8. 自选实战项目

可选方向：

- 库存补货：状态是库存和需求预测，动作是补货量，奖励是利润减去缺货和仓储成本。
- 任务调度：状态是队列和机器负载，动作是选择下一个任务，奖励是负等待时间或吞吐。
- 推荐系统：状态是用户近期行为，动作是推荐内容，奖励是点击、停留、转化的组合。
- 游戏 AI：状态是局面，动作是可行动作，奖励是胜负和中间事件。

## 典型方法定位：AlphaGo / AlphaZero 属于哪一类

AlphaGo 和 AlphaZero 不能简单归为“第二课 planning”或“第三课 Q-learning”。更准确地说，它们是：

```text
已知规则的游戏环境 + 搜索规划 + 自我对弈强化学习 + 神经网络函数逼近
```

它们和第二课相似的地方：

- 围棋规则已知，所以有环境模型。
- 可以用规则模拟“如果在这个局面下这一步，棋盘会怎么变”。
- MCTS（Monte Carlo Tree Search）会在已知规则下向前搜索，这属于 planning / search。

它们和第三课相似的地方：

- 不可能提前穷举所有棋局状态并做完整动态规划。
- 需要通过大量自我对弈产生经验。
- 网络会从经验中学习局面价值和动作倾向。

它们和第三课 Q-learning 不同的地方：

- 不是表格型 `Q(s,a)`。
- 主要学习 policy network `pi(a|s)` 和 value network `V(s)`，而不是维护一个小表格。
- 行动时不是只用 epsilon-greedy，而是用神经网络指导 MCTS 搜索。

一个简化定位：

| 方法 | 模型是否已知 | 是否学习 | 是否搜索/planning | 典型形式 |
| --- | --- | --- | --- | --- |
| 第二课 Dynamic Programming | 已知 | 不从经验学习 | 是 | 小状态空间里直接算 value/policy |
| 第三课 Q-learning | 不要求已知 | 是 | 否 | 从 `(s,a,r,s')` 更新 Q 表 |
| AlphaGo / AlphaZero | 已知游戏规则 | 是，自我对弈 | 是，MCTS | 神经网络 policy/value + 搜索 |

所以它们属于更高级的混合路线：**model-based search + learned policy/value + self-play RL**。

## 推荐阅读代码的顺序

读别人 RL 项目时，不要从模型结构开始。按这个顺序看：

1. 环境定义：`reset()`、`step(action)` 返回什么？
2. reward 设计：奖励是否稀疏？有没有容易被钻空子的地方？
3. 训练循环：每一步收集什么经验？什么时候更新？
4. 算法核心：target 怎么算？loss 是什么？
5. 探索策略：如何选择 action？训练和评估是否不同？
6. 评估方法：是否只看训练奖励？有没有独立测试环境？
