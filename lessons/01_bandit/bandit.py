from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass


@dataclass
class BanditResult:
    name: str
    total_reward: float
    regret: float
    best_action_rate: float
    estimates: list[float]
    counts: list[int]


class GaussianBandit:
    def __init__(self, means: list[float], reward_std: float = 1.0, seed: int = 0):
        self.means = means
        self.reward_std = reward_std
        self.rng = random.Random(seed)
        self.best_mean = max(means)
        self.best_action = means.index(self.best_mean)

    @property
    def n_actions(self) -> int:
        return len(self.means)

    def pull(self, action: int) -> float:
        return self.rng.gauss(self.means[action], self.reward_std)


def choose_greedy(estimates: list[float], rng: random.Random) -> int:
    best_value = max(estimates)
    candidates = [i for i, value in enumerate(estimates) if value == best_value]
    return rng.choice(candidates)


def choose_epsilon_greedy(estimates: list[float], epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(estimates))
    return choose_greedy(estimates, rng)


def choose_ucb(estimates: list[float], counts: list[int], step: int, c: float) -> int:
    for action, count in enumerate(counts):
        if count == 0:
            return action

    scores = [
        estimate + c * math.sqrt(math.log(step + 1) / count)
        for estimate, count in zip(estimates, counts)
    ]
    return scores.index(max(scores))


def run_bandit(
    name: str,
    bandit: GaussianBandit,
    steps: int,
    seed: int,
    epsilon: float = 0.1,
    ucb_c: float = 2.0,
) -> BanditResult:
    rng = random.Random(seed)
    estimates = [0.0 for _ in range(bandit.n_actions)]
    counts = [0 for _ in range(bandit.n_actions)]
    total_reward = 0.0
    best_action_count = 0

    for step in range(1, steps + 1):
        if name == "greedy":
            action = choose_greedy(estimates, rng)
        elif name == "epsilon_greedy":
            action = choose_epsilon_greedy(estimates, epsilon, rng)
        elif name == "ucb":
            action = choose_ucb(estimates, counts, step, ucb_c)
        else:
            raise ValueError(f"Unknown strategy: {name}")

        reward = bandit.pull(action)
        counts[action] += 1
        estimates[action] += (reward - estimates[action]) / counts[action]
        total_reward += reward
        best_action_count += int(action == bandit.best_action)

    optimal_reward = steps * bandit.best_mean
    return BanditResult(
        name=name,
        total_reward=total_reward,
        regret=optimal_reward - total_reward,
        best_action_rate=best_action_count / steps,
        estimates=estimates,
        counts=counts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare exploration strategies on a Gaussian bandit.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of decisions in each run.")
    parser.add_argument("--runs", type=int, default=100, help="Number of independent runs to average.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate for epsilon-greedy.")
    parser.add_argument("--reward-std", type=float, default=1.0, help="Reward noise standard deviation.")
    parser.add_argument("--ucb-c", type=float, default=2.0, help="Exploration strength for UCB.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    true_means = [0.2, 0.0, 1.4, 0.7, 1.0]

    print("True action values:", [round(v, 2) for v in true_means])
    print(f"Best action: {true_means.index(max(true_means))}\n")
    print(f"Averaged over {args.runs} independent runs, {args.steps} steps each.")
    print(f"epsilon={args.epsilon}, reward_std={args.reward_std}, ucb_c={args.ucb_c}\n")

    for strategy in ["greedy", "epsilon_greedy", "ucb"]:
        results = []
        for run in range(args.runs):
            bandit = GaussianBandit(true_means, reward_std=args.reward_std, seed=10_000 + run)
            results.append(
                run_bandit(
                    strategy,
                    bandit,
                    steps=args.steps,
                    seed=20_000 + run,
                    epsilon=args.epsilon,
                    ucb_c=args.ucb_c,
                )
            )

        avg_reward = sum(result.total_reward for result in results) / args.runs
        avg_regret = sum(result.regret for result in results) / args.runs
        avg_best_action_rate = sum(result.best_action_rate for result in results) / args.runs
        final_result = results[-1]

        print(f"Strategy: {strategy}")
        print(f"  avg total reward     : {avg_reward:8.2f}")
        print(f"  avg regret           : {avg_regret:8.2f}")
        print(f"  avg best action rate : {avg_best_action_rate:8.2%}")
        print(f"  last-run estimates   : {[round(v, 2) for v in final_result.estimates]}")
        print(f"  last-run counts      : {final_result.counts}\n")


if __name__ == "__main__":
    main()
