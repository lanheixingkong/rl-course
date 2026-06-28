# Project Instructions

## Course Goal

This project is an iterative, practice-first RL course for the user.

The user wants to learn and use reinforcement learning without following a purely theoretical textbook style. The course should teach RL through practical cases, runnable code, and only the amount of theory needed to understand and generalize the method.

End goal:

- The user can read and understand RL methods used by others.
- The user can decide whether a new real-world problem is suitable for RL.
- The user can model a new problem with states, actions, rewards, transitions, and episode boundaries.
- The user can build a baseline RL solution and iterate toward stronger methods.

## Working Style

- Do not try to fully finalize the entire course upfront.
- Keep only a lightweight roadmap, then revise and expand lessons as the user learns.
- Prefer small runnable examples over long theoretical explanations.
- The user should read non-code files too. Root `README.md` gives the course route, lesson `README.md` gives the entry point and commands, `docs/articles/...` gives the complete shareable tutorial, and `docs/study_plan.md` gives the current operational steps and pass criteria.
- After the user confirms a shareable lesson tutorial is complete, generate a matching cover image for that article and save it in the same directory as the article. Prefer a wide blog-cover ratio of 2.35:1 unless the user asks otherwise.
- Each lesson should combine:
  - a practical problem,
  - executable code,
  - observations from running the code,
  - the minimal RL principle behind it,
  - small exercises that modify the example.
- Use Chinese for course explanations unless the user asks otherwise.
- Keep code readable and beginner-friendly; avoid unnecessary abstraction.

## Current Course State

Initial materials created:

- `README.md`: course overview and lightweight roadmap.
- `docs/study_plan.md`: current operational learning plan and pass criteria.
- `docs/curriculum.md`: broader curriculum notes and RL problem-modeling template.
- `lessons/01_bandit/`: multi-armed bandit example.
- `lessons/02_gridworld_dp/`: GridWorld dynamic programming example.
- `lessons/03_q_learning/`: Q-learning GridWorld example.
- `requirements.txt`: dependencies for later lessons; first three lessons use only Python standard library.

Repository remote: `https://github.com/lanheixingkong/rl-course.git`. Lesson 01 and Lesson 02 have been committed and pushed to `main`. Lesson 03 has been completed by the user and is ready to commit/push.

## Recommended Next Step

After committing/pushing Lesson 03, continue with the next lesson only after the user asks to start it.

Lesson 01 has been started. Current artifacts:

- `lessons/01_bandit/bandit.py` supports CLI parameters: `--steps`, `--runs`, `--epsilon`, `--reward-std`, `--ucb-c`.
- `lessons/01_bandit/README.md` is the lesson entry point.
- `lessons/01_bandit/README.md` now includes beginner-friendly output explanation immediately after the run command, including meanings of `greedy`, `epsilon_greedy`, `ucb`, `avg total reward`, `avg regret`, `avg best action rate`, `last-run estimates`, and `last-run counts`.
- Clarification added for a beginner confusion: `5 actions` means five available choices at every decision step; `2000 steps` means the agent repeatedly chooses one of those same five actions 2000 times, not that there are 2000 different actions.
- Lesson 01 README now explains parameter experiments in beginner terms: `epsilon` tests too little/too much exploration, `steps` tests short vs long horizons, `reward-std` tests noisy feedback, and `ucb-c` tests UCB exploration strength. Each parameter section should explain what to look at and what question it validates.
- The shareable lesson article `docs/articles/lesson_01_bandit_tutorial.md` has been synced with beginner explanations from the README: difference between action count and steps, output metric meanings, `last-run estimates/counts`, and the purpose of each parameter experiment.
- Lesson 01 README now teaches formulas through code-first comparison. Continue this pattern: explain code variables first, then map them to formulas. For example `estimates[action]` maps to `Q(a)`, `counts[action]` maps to `N(a)`, `reward` maps to `r`, and `true_means` maps to `q*(a)`.
- Lesson 01 README now includes detailed implementation explanations for all three strategies: `choose_greedy`, `choose_epsilon_greedy`, and `choose_ucb`. Future lessons should explain algorithm functions line-by-line before moving to formulas.
- UCB explanation has been expanded: present UCB as a family of methods based on `estimated reward + uncertainty bonus`, explain the roles of `count`, `log(step + 1)`, `sqrt`, and `c`, and clarify that the code uses a UCB1-style formula rather than the only possible UCB formula.
- Clarified that `Q(a) <- Q(a) + (r - Q(a)) / N(a)` is not a universal RL formula. It is the incremental sample-average update used in this bandit lesson. The broader pattern is `new estimate = old estimate + step size * error`; later Q-learning uses learning rate `alpha` and TD error.
- `docs/articles/lesson_01_bandit_tutorial.md` is the shareable Chinese tutorial for the lesson.
- `docs/study_plan.md` records how the user should study now, including Lesson 01 steps and pass criteria.
- The tutorial now includes a conceptual comparison for users with ML/DL background: why bandit is an RL entry problem, when supervised ML/neural networks can solve parts of it, and why exploration/partial feedback/policy-driven data collection make it an RL-style problem.
- Important clarification recorded: the distinction is not merely whether data exists, but whether data is complete, sufficiently covers actions, has controllable bias, and whether the deployed policy changes future data. Prefer traditional ML when data is reliable and fixed; prefer Bandit/RL when actions generate partial feedback and the agent must learn while collecting data.

Lesson 02 preparation has started. Current artifacts:

- `lessons/02_gridworld_dp/gridworld_dp.py` now supports CLI parameters: `--gamma`, `--step-reward`, `--goal-reward`, `--pit-reward`.
- `lessons/02_gridworld_dp/README.md` has been rewritten as a beginner-friendly lesson entry point.
- `docs/articles/lesson_02_gridworld_dp_tutorial.md` is the shareable Chinese tutorial for Lesson 02.
- `docs/study_plan.md` now has Lesson 02 steps and pass criteria.

Lesson 02 should follow the Lesson 01 teaching standard:

- Explain output immediately after the run command.
- Explain `Values` and `Policy` before formulas.
- Explain each parameter experiment in terms of what question it validates.
- Explain code variables before formulas.
- When explaining source code, first provide a source-code map and execution flow. Do not jump into isolated code snippets. For each highlighted snippet, state where it is in the source file, what larger function/flow it belongs to, why this snippet is being discussed, and what the surrounding non-highlighted code does.
- Map `values[state]` to `V(s)`, `reward` to `r`, `next_state` to `s'`, and `gamma` to the discount factor.
- Explain policy evaluation and policy improvement through their code blocks before using Bellman terminology.
- Explain two Lesson 02 beginner pitfalls explicitly: changing parameters may change `Values` without changing `Policy` in this small deterministic map; positive `step_reward` is not universally invalid, but can make cycling more valuable than terminal completion and cause non-convergence/repeating policies.
- Clarify in Lesson 02 that the agent directly maximizes cumulative reward, not the natural-language goal "reach the terminal"; reaching the goal is induced by reward design (`goal_reward`, `pit_reward`, `step_reward`). The reward values are visible in config output and implemented in `GridWorld.step`.
- Lesson 02 code now supports `--show-evaluation` and `--debug-sweeps N` to print the first N sweeps inside `evaluate_policy()`. Use this when the user wants to see how `values[state] = reward + gamma * values[next_state]` is computed with concrete `state`, `action`, `next_state`, `reward`, `old V`, `new V`, and `delta` columns.
- When explaining why self-consistent values can evaluate a policy, anchor on the definition of `V(s)`: discounted return from state `s` under the fixed policy. The return decomposes recursively into immediate `reward + gamma * value(next_state)`, so the correct value function must satisfy `V(s) = r + gamma * V(s')` in this deterministic lesson. Explain that iteration works because `gamma < 1` discounts future influence, so repeated updates converge toward the fixed point.
- When explaining why `evaluate_policy()` needs many sweeps, clarify that sweeps are repeated value recomputations under a fixed policy, not real environment steps or trial-and-error. The process is iterative policy evaluation: values are updated until they are self-consistent. Use the `(1,0)` self-loop example from the initial all-right policy: `V(1,0) <- -0.04 + 0.95 * V(1,0)`, which slowly approaches `-0.8`; high `gamma` and small `theta` make convergence slower.
- Clarify why Lesson 02 uses iterative value computation instead of "walking the maze": because the environment model is known through `env.step(state, action)`, planning can compute expected results directly. Trial-and-error/trajectory sampling is useful when the model is unknown, but it costs samples, introduces estimation noise, and may be expensive or risky in real systems. Bridge this distinction to Lesson 03 Q-learning, where the agent learns from experienced `(state, action, reward, next_state)` transitions instead of planning from a known model.
- When explaining reward design, emphasize the RL interface: agent selects action, environment returns next_state and scalar reward, and the algorithm optimizes discounted return `r1 + gamma*r2 + ...`; natural-language goals must be encoded as rewards.
- Clarify the Lesson 02 `planning` assumption: planning means the environment model is known, i.e. the agent can compute `next_state` and `reward` for a `(state, action)` pair. The counterpart is not only "try first, then plan"; model-unknown settings include model-free RL (learn values/policy directly from experience, e.g. Q-learning), model-based RL (learn a model then plan), and offline RL (learn from historical interaction data). Use real-world examples such as known-map path planning, warehouse robots, chess/game rules for planning; online recommendation, game agents, and robot simulation for model-free RL; autonomous driving/robot dynamics/industrial control for model-based RL; and medical/finance/driving logs for offline RL.

Suggested Lesson 02 flow:

1. Ask the user to run `python lessons/02_gridworld_dp/gridworld_dp.py`.
2. Interpret `Values` and `Policy`.
3. Have the user change `gamma`, `step-reward`, and `pit-reward`.
4. Read `GridWorld.step`, `evaluate_policy`, `improve_policy`, and `policy_iteration`.
5. Only after the code is clear, introduce `V(s)` and the simplified Bellman update `V(s) <- r + gamma * V(s')`.
6. Before moving to Lesson 03, ask the user to explain the difference between policy evaluation and policy improvement.

Lesson 03 has been completed. Current artifacts:

- `lessons/03_q_learning/q_learning_gridworld.py` is now a command-line Q-learning GridWorld example.
- It supports CLI parameters: `--episodes`, `--max-steps`, `--alpha`, `--gamma`, `--epsilon`, `--step-reward`, `--goal-reward`, `--pit-reward`, `--slip-probability`, `--seed`, `--log-every`, and `--debug-episodes`.
- `--debug-episodes N` prints step-by-step Q updates for the first N episodes, including `state`, `action`, `reward`, `next`, `old Q`, `target`, `td err`, and `new Q`.
- `lessons/03_q_learning/README.md` has been rewritten as a beginner-friendly lesson entry point.
- `docs/articles/lesson_03_q_learning_tutorial.md` is the shareable Chinese tutorial for Lesson 03.
- `docs/articles/lesson_03_q_learning_cover.png` is the cover image for Lesson 03.
- `docs/study_plan.md` now has Lesson 03 steps and pass criteria.

Lesson 03 should follow the established teaching standard:

- Explain output immediately after the run command.
- Explain `episode`, `avg return`, `success rate`, `Best Q value`, and `Greedy policy` before formulas.
- Explain the difference between Lesson 02 planning and Lesson 03 model-free learning: Q-learning still calls `env.step()` to receive experience, but it does not use a known model to plan over all state-action outcomes.
- When users are confused by "known complete environment model" vs "learning from one experience", use the concrete GridWorld example: Lesson 02 can query/compute all possible results from `(2,0)` for `U/R/D/L` without walking and then plan globally; Lesson 03 only updates the action actually taken, e.g. after experiencing `(2,0), R, -0.04, (2,1)`, it updates `Q((2,0), R)` only. Stress that the difference is algorithmic usage of `env.step()`, not whether the Python simulation contains an environment function.
- Explain `V(s)` vs `Q(s,a)` before introducing the Q-learning update.
- Explain source code by execution flow: `main()` -> `train_q_learning()` -> `choose_action()` -> `env.step()` -> Q update -> final printing.
- Map `q[state][action]` to `Q(s,a)`, `reward` to `r`, `next_state` to `s'`, `alpha` to learning rate, `gamma` to discount factor, and `max(q[next_state].values())` to `max_a' Q(s',a')`.
- Teach the update as `new estimate = old estimate + step size * error` before showing the full formula.
- Clarify `target = reward + gamma * next_best` and `td_error = target - old_q`.
- When explaining Lesson 03 debug rows, stress that Q values are keyed by `(state, action)`, not just `state`. For example, step1 may update `Q((2,0), L)` to `-0.01`, while step2 reads `Q((2,0), R)`, whose old value is still `0.00`. Explain `target` as `reward + gamma * next_best`; early in training `next_best` is often `0.00`, so `target` may equal the immediate `reward`. If the move is non-terminal, that reward usually comes from `step_reward`, but `step_reward`, `reward`, and `target` are distinct concepts.
- Clarify epsilon-greedy: `epsilon` controls explicit random exploration, but `epsilon=0` may still have randomness early because the code breaks ties randomly among equal Q values.
- Clarify why Q-learning is off-policy by first explaining why the concept matters: it answers how Q-learning can behave with epsilon-greedy exploration while learning toward a greedy target policy. Behavior uses epsilon-greedy, while the update target uses greedy `max` over next-state Q values. Mention that this distinction helps later compare Q-learning/DQN with Sarsa and PPO.
- Conceptual clarification: AlphaGo/AlphaZero are neither simply Lesson 02 planning nor Lesson 03 Q-learning. They are better described as known-game-rule model-based search/planning (MCTS) plus self-play RL plus neural network function approximation. They resemble Lesson 02 because game rules are known and MCTS plans/searches with the model; they resemble Lesson 03 because policy/value estimates are learned from self-play experience; they differ from tabular Q-learning because they learn policy/value networks and use search rather than maintaining a small `Q(s,a)` table.

Suggested Lesson 03 flow:

1. Ask the user to run `python lessons/03_q_learning/q_learning_gridworld.py`.
2. Interpret training logs, `Best Q value`, and `Greedy policy`.
3. Ask the user to run `python lessons/03_q_learning/q_learning_gridworld.py --episodes 5 --debug-episodes 1 --log-every 0`.
4. Interpret one Q update row before showing the formula.
5. Have the user change `episodes`, `epsilon`, `alpha`, and `slip-probability`.
6. Read `make_q_table`, `choose_action`, and the Q update block inside `train_q_learning`.
7. Before moving on, ask the user to explain `target`, `td_error`, model-free, and off-policy in their own words.

## Environment Notes

- Python environments are managed with `conda`.
- Python dependencies should be installed with `uv pip`, not plain `pip`.
- Node.js environments are managed with `nvm`.
- Internet access may require local Clash proxy at `127.0.0.1:7890`.
