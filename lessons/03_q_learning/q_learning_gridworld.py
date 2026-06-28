from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass


Action = str
State = tuple[int, int]

ACTIONS: list[Action] = ["U", "R", "D", "L"]
DELTAS: dict[Action, State] = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}


@dataclass(frozen=True)
class GridWorld:
    rows: int
    cols: int
    start: State
    terminal_rewards: dict[State, float]
    walls: set[State]
    step_reward: float = -0.04
    slip_probability: float = 0.0

    def states(self) -> list[State]:
        return [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if (row, col) not in self.walls
        ]

    def is_terminal(self, state: State) -> bool:
        return state in self.terminal_rewards

    def reset(self) -> State:
        return self.start

    def step(self, state: State, action: Action, rng: random.Random) -> tuple[State, float, bool]:
        actual_action = action
        if rng.random() < self.slip_probability:
            actual_action = rng.choice(ACTIONS)

        row_delta, col_delta = DELTAS[actual_action]
        next_state = (state[0] + row_delta, state[1] + col_delta)

        out_of_bounds = not (0 <= next_state[0] < self.rows and 0 <= next_state[1] < self.cols)
        if out_of_bounds or next_state in self.walls:
            next_state = state

        reward = self.terminal_rewards.get(next_state, self.step_reward)
        done = self.is_terminal(next_state)
        return next_state, reward, done


def make_q_table(env: GridWorld) -> dict[State, dict[Action, float]]:
    return {
        state: {action: 0.0 for action in ACTIONS}
        for state in env.states()
        if not env.is_terminal(state)
    }


def choose_action(
    q: dict[State, dict[Action, float]],
    state: State,
    epsilon: float,
    rng: random.Random,
) -> Action:
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)

    values = q[state]
    best_value = max(values.values())
    best_actions = [action for action, value in values.items() if value == best_value]
    return rng.choice(best_actions)


def train_q_learning(
    env: GridWorld,
    episodes: int = 5000,
    max_steps: int = 100,
    alpha: float = 0.2,
    gamma: float = 0.95,
    epsilon: float = 0.2,
    seed: int = 0,
    log_every: int = 500,
    debug_episodes: int = 0,
) -> tuple[dict[State, dict[Action, float]], list[float], list[int], list[bool]]:
    rng = random.Random(seed)
    q = make_q_table(env)
    returns = []
    episode_lengths = []
    successes = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False

        if episode <= debug_episodes:
            print(f"Episode {episode}:")
            print("  step  state   action  reward  next    old Q   target  td err  new Q")

        for step_index in range(1, max_steps + 1):
            action = choose_action(q, state, epsilon, rng)
            next_state, reward, done = env.step(state, action, rng)
            total_reward += reward

            old_q = q[state][action]
            next_best = 0.0 if done else max(q[next_state].values())
            target = reward + gamma * next_best
            td_error = target - old_q
            q[state][action] = old_q + alpha * td_error

            if episode <= debug_episodes:
                print(
                    f"  {step_index:>4}  {state!s:7} {action:^6} {reward:6.2f} "
                    f"{next_state!s:7} {old_q:7.2f} {target:7.2f} {td_error:7.2f} {q[state][action]:7.2f}"
                )

            state = next_state
            if done:
                break

        returns.append(total_reward)
        episode_lengths.append(step_index)
        successes.append(done and state == max(env.terminal_rewards, key=env.terminal_rewards.get))

        if episode <= debug_episodes:
            status = "success" if successes[-1] else ("terminal" if done else "max steps")
            print(f"  total reward = {total_reward:.2f}, steps = {step_index}, end = {status}\n")

        if log_every > 0 and episode % log_every == 0:
            recent_returns = returns[-100:]
            recent_lengths = episode_lengths[-100:]
            recent_successes = successes[-100:]
            avg_return = sum(recent_returns) / len(recent_returns)
            avg_steps = sum(recent_lengths) / len(recent_lengths)
            success_rate = sum(recent_successes) / len(recent_successes)
            print(
                f"episode {episode:5d} | "
                f"avg return last 100: {avg_return:6.3f} | "
                f"avg steps: {avg_steps:5.1f} | "
                f"success rate: {success_rate:5.2f}"
            )

    return q, returns, episode_lengths, successes


def print_best_values(env: GridWorld, q: dict[State, dict[Action, float]]) -> None:
    print("\nBest Q value by state:")
    for row in range(env.rows):
        cells = []
        for col in range(env.cols):
            state = (row, col)
            if state in env.walls:
                cells.append("  WALL ")
            elif env.is_terminal(state):
                cells.append(f"{env.terminal_rewards[state]:6.2f}")
            else:
                cells.append(f"{max(q[state].values()):6.2f}")
        print(" ".join(cells))


def print_greedy_policy(env: GridWorld, q: dict[State, dict[Action, float]]) -> None:
    print("\nGreedy policy:")
    for row in range(env.rows):
        cells = []
        for col in range(env.cols):
            state = (row, col)
            if state in env.walls:
                cells.append("WALL")
            elif env.is_terminal(state):
                cells.append(f"{env.terminal_rewards[state]:+4.0f}")
            else:
                values = q[state]
                best_action = max(values, key=values.get)
                cells.append(f"  {best_action} ")
        print(" ".join(cells))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn a GridWorld policy with Q-learning.")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps in one episode.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Learning rate for Q updates.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor for future rewards.")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Exploration probability for epsilon-greedy.")
    parser.add_argument("--step-reward", type=float, default=-0.04, help="Reward received on each non-terminal move.")
    parser.add_argument("--goal-reward", type=float, default=1.0, help="Reward for reaching the goal state.")
    parser.add_argument("--pit-reward", type=float, default=-1.0, help="Reward for reaching the pit state.")
    parser.add_argument("--slip-probability", type=float, default=0.0, help="Probability that an action becomes random.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--log-every", type=int, default=500, help="Print training progress every N episodes; 0 disables logs.")
    parser.add_argument("--debug-episodes", type=int, default=0, help="Print step-by-step Q updates for the first N episodes.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes < 1:
        print("Error: --episodes must be >= 1.", file=sys.stderr)
        raise SystemExit(2)
    if args.max_steps < 1:
        print("Error: --max-steps must be >= 1.", file=sys.stderr)
        raise SystemExit(2)
    if not (0 < args.alpha <= 1):
        print("Error: --alpha must be in (0, 1].", file=sys.stderr)
        raise SystemExit(2)
    if not (0 <= args.gamma < 1):
        print("Error: --gamma must be in [0, 1).", file=sys.stderr)
        raise SystemExit(2)
    if not (0 <= args.epsilon <= 1):
        print("Error: --epsilon must be in [0, 1].", file=sys.stderr)
        raise SystemExit(2)
    if not (0 <= args.slip_probability <= 1):
        print("Error: --slip-probability must be in [0, 1].", file=sys.stderr)
        raise SystemExit(2)
    if args.log_every < 0:
        print("Error: --log-every must be >= 0.", file=sys.stderr)
        raise SystemExit(2)
    if args.debug_episodes < 0:
        print("Error: --debug-episodes must be >= 0.", file=sys.stderr)
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    validate_args(args)

    env = GridWorld(
        rows=3,
        cols=4,
        start=(2, 0),
        terminal_rewards={(0, 3): args.goal_reward, (1, 3): args.pit_reward},
        walls={(1, 1)},
        step_reward=args.step_reward,
        slip_probability=args.slip_probability,
    )

    print("Q-learning GridWorld config:")
    print(f"  episodes         : {args.episodes}")
    print(f"  max_steps        : {args.max_steps}")
    print(f"  alpha            : {args.alpha}")
    print(f"  gamma            : {args.gamma}")
    print(f"  epsilon          : {args.epsilon}")
    print(f"  step_reward      : {args.step_reward}")
    print(f"  goal_reward      : {args.goal_reward}")
    print(f"  pit_reward       : {args.pit_reward}")
    print(f"  slip_probability : {args.slip_probability}")
    print(f"  seed             : {args.seed}")
    print()

    q, returns, episode_lengths, successes = train_q_learning(
        env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
        log_every=args.log_every,
        debug_episodes=args.debug_episodes,
    )

    recent_window = min(100, len(returns))
    avg_return = sum(returns[-recent_window:]) / recent_window
    avg_steps = sum(episode_lengths[-recent_window:]) / recent_window
    success_rate = sum(successes[-recent_window:]) / recent_window
    print("\nTraining summary:")
    print(f"  avg return over last {recent_window}: {avg_return:.3f}")
    print(f"  avg steps over last {recent_window} : {avg_steps:.1f}")
    print(f"  success rate last {recent_window}   : {success_rate:.2f}")

    print_best_values(env, q)
    print_greedy_policy(env, q)


if __name__ == "__main__":
    main()
