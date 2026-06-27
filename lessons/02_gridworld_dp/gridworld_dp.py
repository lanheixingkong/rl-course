from __future__ import annotations

import argparse
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


class PolicyIterationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GridWorld:
    rows: int
    cols: int
    terminal_rewards: dict[State, float]
    walls: set[State]
    step_reward: float = -0.04

    def states(self) -> list[State]:
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.walls
        ]

    def is_terminal(self, state: State) -> bool:
        return state in self.terminal_rewards

    def step(self, state: State, action: Action) -> tuple[State, float]:
        if self.is_terminal(state):
            return state, 0.0

        dr, dc = DELTAS[action]
        next_state = (state[0] + dr, state[1] + dc)

        out_of_bounds = not (0 <= next_state[0] < self.rows and 0 <= next_state[1] < self.cols)
        if out_of_bounds or next_state in self.walls:
            next_state = state

        reward = self.terminal_rewards.get(next_state, self.step_reward)
        return next_state, reward


def evaluate_policy(
    env: GridWorld,
    policy: dict[State, Action],
    gamma: float,
    theta: float = 1e-4,
    show_evaluation: bool = False,
    max_debug_sweeps: int = 3,
    policy_iteration_number: int = 1,
) -> dict[State, float]:
    values = {state: 0.0 for state in env.states()}

    sweep = 0
    while True:
        sweep += 1
        delta = 0.0
        debug_rows = []
        for state in env.states():
            old_value = values[state]
            if env.is_terminal(state):
                values[state] = 0.0
            else:
                action = policy[state]
                next_state, reward = env.step(state, action)
                values[state] = reward + gamma * values[next_state]
                if show_evaluation and sweep <= max_debug_sweeps:
                    debug_rows.append((state, action, next_state, reward, old_value, values[state]))
            delta = max(delta, abs(old_value - values[state]))
        if show_evaluation and sweep <= max_debug_sweeps:
            print_evaluation_sweep(policy_iteration_number, sweep, debug_rows, delta)
        if delta < theta:
            if show_evaluation:
                if sweep > max_debug_sweeps:
                    print(
                        f"  ... evaluation converged after {sweep} sweeps "
                        f"(only first {max_debug_sweeps} shown)."
                    )
                else:
                    print(f"  evaluation converged after {sweep} sweeps.")
                print()
            return values


def improve_policy(
    env: GridWorld,
    values: dict[State, float],
    policy: dict[State, Action],
    gamma: float,
) -> tuple[dict[State, Action], bool]:
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


def policy_iteration(
    env: GridWorld,
    gamma: float = 0.95,
    show_evaluation: bool = False,
    max_debug_sweeps: int = 3,
) -> tuple[dict[State, Action], dict[State, float], int]:
    policy = {
        state: "R"
        for state in env.states()
        if not env.is_terminal(state)
    }
    seen_policies = set()

    iterations = 0
    while True:
        iterations += 1
        if show_evaluation:
            print(f"Policy iteration {iterations}: evaluate current policy")
        values = evaluate_policy(
            env,
            policy,
            gamma,
            show_evaluation=show_evaluation,
            max_debug_sweeps=max_debug_sweeps,
            policy_iteration_number=iterations,
        )
        policy, stable = improve_policy(env, values, policy, gamma)
        if stable:
            return policy, values, iterations
        signature = tuple(sorted(policy.items()))
        if signature in seen_policies:
            raise PolicyIterationError(
                "Policy iteration detected a repeating policy. This often means the reward design makes "
                "cycling attractive, for example a positive step_reward can reward the agent for walking "
                "forever instead of reaching a terminal state."
            )
        seen_policies.add(signature)


def print_evaluation_sweep(
    policy_iteration_number: int,
    sweep: int,
    debug_rows: list[tuple[State, Action, State, float, float, float]],
    delta: float,
) -> None:
    print(f"  evaluation sweep {sweep} in policy iteration {policy_iteration_number}")
    print("    state   action  next    reward   old V    new V")
    for state, action, next_state, reward, old_value, new_value in debug_rows:
        print(
            f"    {state!s:7} {action:^6} {next_state!s:7} "
            f"{reward:7.2f} {old_value:7.2f} {new_value:7.2f}"
        )
    print(f"    max value change delta = {delta:.6f}")
    print()


def print_values(env: GridWorld, values: dict[State, float]) -> None:
    print("Values:")
    for r in range(env.rows):
        cells = []
        for c in range(env.cols):
            state = (r, c)
            if state in env.walls:
                cells.append("  WALL ")
            elif env.is_terminal(state):
                cells.append(f"{env.terminal_rewards[state]:6.2f}")
            else:
                cells.append(f"{values[state]:6.2f}")
        print(" ".join(cells))
    print()


def print_policy(env: GridWorld, policy: dict[State, Action]) -> None:
    print("Policy:")
    for r in range(env.rows):
        cells = []
        for c in range(env.cols):
            state = (r, c)
            if state in env.walls:
                cells.append("WALL")
            elif env.is_terminal(state):
                cells.append(f"{env.terminal_rewards[state]:+4.0f}")
            else:
                cells.append(f"  {policy[state]} ")
        print(" ".join(cells))
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve a small GridWorld with policy iteration.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor for future rewards.")
    parser.add_argument("--step-reward", type=float, default=-0.04, help="Reward received on each non-terminal move.")
    parser.add_argument("--goal-reward", type=float, default=1.0, help="Reward for reaching the goal state.")
    parser.add_argument("--pit-reward", type=float, default=-1.0, help="Reward for reaching the pit state.")
    parser.add_argument(
        "--show-evaluation",
        action="store_true",
        help="Print the first few value-update sweeps inside evaluate_policy().",
    )
    parser.add_argument(
        "--debug-sweeps",
        type=int,
        default=3,
        help="Number of evaluation sweeps to print when --show-evaluation is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 <= args.gamma < 1):
        print("Error: --gamma must be in [0, 1) for this lesson's iterative value updates.", file=sys.stderr)
        raise SystemExit(2)
    if args.debug_sweeps < 1:
        print("Error: --debug-sweeps must be >= 1.", file=sys.stderr)
        raise SystemExit(2)

    env = GridWorld(
        rows=3,
        cols=4,
        terminal_rewards={(0, 3): args.goal_reward, (1, 3): args.pit_reward},
        walls={(1, 1)},
        step_reward=args.step_reward,
    )

    try:
        policy, values, iterations = policy_iteration(
            env,
            gamma=args.gamma,
            show_evaluation=args.show_evaluation,
            max_debug_sweeps=args.debug_sweeps,
        )
    except PolicyIterationError as exc:
        print("Policy iteration did not converge to a stable policy.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print(file=sys.stderr)
        print("Try one of these:", file=sys.stderr)
        print("  - use a non-positive step reward, e.g. --step-reward -0.04", file=sys.stderr)
        print("  - lower the positive step reward below the value of reaching the goal", file=sys.stderr)
        print("  - redesign the task with a finite horizon if you want to reward each step", file=sys.stderr)
        raise SystemExit(1)

    print("GridWorld config:")
    print(f"  gamma       : {args.gamma}")
    print(f"  step_reward : {args.step_reward}")
    print(f"  goal_reward : {args.goal_reward}")
    print(f"  pit_reward  : {args.pit_reward}")
    print()
    print("Legend:")
    print("  U/R/D/L = move up/right/down/left")
    print(f"  {args.goal_reward:+g}      = goal terminal state")
    print(f"  {args.pit_reward:+g}      = pit terminal state")
    print("  WALL    = blocked cell")
    print()
    print(f"Policy iteration converged in {iterations} iterations.\n")
    print_values(env, values)
    print_policy(env, policy)


if __name__ == "__main__":
    main()
