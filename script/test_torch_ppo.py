import argparse
import time

import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn, optim
from torch.distributions import Categorical as CategoricalDist
from torchrl.collectors import Collector
from torchrl.data import Bounded, Composite
from torchrl.data import Categorical as CategoricalSpec
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

# ------------------------
# Custom Dice Environment
# ------------------------


class DiceEnv(EnvBase):
    def __init__(self, num_dice: int = 6):
        super().__init__()
        self.num_dice = num_dice
        self.max_steps = num_dice
        self.step_count = 0

        # Weighted dice probabilities (faces 0-5)
        self.weights = torch.tensor([0.05, 0.1, 0.2, 0.25, 0.2, 0.2])

        # Observation spec
        self.observation_spec = Composite(
            dice=Bounded(low=0, high=5, shape=(num_dice,), dtype=torch.int64),
            rolled=Bounded(low=0, high=1, shape=(num_dice,), dtype=torch.int64),
        )

        # Action spec (choose die 0 to num_dice-1)
        self.action_spec = CategoricalSpec(n=num_dice)

        self.reward_spec = Bounded(low=0, high=num_dice, shape=(1,))

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            torch.manual_seed(seed)

    def to(self, device):
        super().to(device)
        self.weights = self.weights.to(device)
        return self

    def _get_device(self):
        dev = self.device
        return dev if dev is not None else torch.device("cpu")

    def _reset(self, tensordict=None):
        device = self._get_device()
        self.step_count = 0
        self.dice = torch.zeros(self.num_dice, dtype=torch.int64, device=device)
        self.rolled = torch.zeros(self.num_dice, dtype=torch.int64, device=device)
        return TensorDict(
            {"dice": self.dice, "rolled": self.rolled},
            batch_size=[],
            device=device,
        )

    def _step(self, tensordict):
        device = self._get_device()
        action = tensordict["action"].item()

        if self.rolled[action] == 0:
            face = torch.multinomial(self.weights.to(device), 1)
            self.dice[action] = face.squeeze()
            self.rolled[action] = 1

        self.step_count += 1
        done = self.step_count >= self.max_steps

        if done:
            reward = torch.tensor(
                [len(torch.unique(self.dice))], dtype=torch.float32, device=device
            )
        else:
            reward = torch.tensor([0.0], device=device)

        return TensorDict(
            {
                "dice": self.dice.clone(),
                "rolled": self.rolled.clone(),
                "reward": reward,
                "done": torch.tensor(done, device=device),
            },
            batch_size=[],
            device=device,
        )


# ------------------------
# Policy Network
# ------------------------


class PolicyNet(nn.Module):
    def __init__(self, obs_size: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_actions),
        )

    def forward(self, dice, rolled):
        x = torch.cat([dice.float(), rolled.float()], dim=-1)
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, obs_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, dice, rolled):
        x = torch.cat([dice.float(), rolled.float()], dim=-1)
        return self.net(x)


def run_training(device: torch.device, verbose: bool = True) -> tuple[list[float], list, list]:
    """Run full training pipeline on given device. Returns (batch_times, scores_before, scores_after)."""
    learning_rate = 3e-4
    gamma = 0.99
    lmbda = 0.95
    clip_epsilon = 0.2
    num_dice = 6
    frames_per_batch = 256
    total_batches = 10
    total_frames = frames_per_batch * total_batches

    env = DiceEnv(num_dice=num_dice)
    env = env.to(device)
    check_env_specs(env)

    obs_size = num_dice * 2
    policy_net = PolicyNet(obs_size=obs_size, num_actions=num_dice)
    value_net = ValueNet(obs_size=obs_size)

    policy_module = TensorDictModule(
        policy_net,
        in_keys=["dice", "rolled"],
        out_keys=["logits"],
    )

    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        distribution_class=CategoricalDist,
        return_log_prob=True,
    )

    critic = ValueOperator(module=value_net, in_keys=["dice", "rolled"])
    loss_module = ClipPPOLoss(actor, critic, clip_epsilon=clip_epsilon, entropy_bonus=True)
    loss_module = loss_module.to(device)

    advantage_module = GAE(gamma=gamma, lmbda=lmbda, value_network=critic)

    collector = Collector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
    )

    optimizer = optim.Adam(loss_module.parameters(), lr=learning_rate)

    def run_eval(env, actor, n_games=100):
        scores = []
        actor.eval()
        with torch.no_grad():
            for _ in range(n_games):
                td = env.reset()
                while True:
                    td = actor(td)
                    td = env.step(td)
                    if td["next"]["done"].item():
                        scores.append(td["next"]["reward"].item())
                        break
                    td = td["next"]
        return scores

    torch.manual_seed(42)
    scores_before = run_eval(env, actor)

    batch_idx = 0
    actor.train()

    batch_times: list[float] = []
    for batch in collector:
        t0 = time.perf_counter()
        batch = advantage_module(batch)
        loss = loss_module(batch)
        optimizer.zero_grad()
        loss["loss_objective"].backward()
        optimizer.step()
        elapsed = time.perf_counter() - t0
        batch_times.append(elapsed)

        if verbose:
            loss_value = loss["loss_objective"].item()
            parts = []
            for k, v in loss.items():
                if k != "loss_objective" and isinstance(v, torch.Tensor) and v.numel() == 1:
                    parts.append(f"{k}={v.item():.4f}")
            extra = f"  [{', '.join(parts)}]" if parts else ""
            print(f"Batch {batch_idx}: {elapsed:.3f}s  loss={loss_value:.4f}{extra}")
        batch_idx += 1

    scores_after = run_eval(env, actor)
    return batch_times, scores_before, scores_after


def main():
    parser = argparse.ArgumentParser(description="PPO training on dice game")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU (default: GPU)")
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
        print("Using device: CPU")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda:0")
        print(f"Using device: {device}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    batch_times, scores_before, scores_after = run_training(device, verbose=True)
    total = sum(batch_times)
    print(f"\nTotal iteration time: {total:.2f}s  (avg {total/len(batch_times):.3f}s/batch)")

    num_dice = 6

    # ------------------------
    # Combined histogram figure
    # ------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    bins = range(1, num_dice + 2)
    ax1.hist(scores_before, bins=bins, align="left", rwidth=0.8, edgecolor="black")
    ax1.set_xlabel("Score (unique dice count)")
    ax1.set_ylabel("Games")
    ax1.set_title("Before training")
    ax1.set_xticks(range(1, num_dice + 1))
    ax2.hist(scores_after, bins=bins, align="left", rwidth=0.8, edgecolor="black")
    ax2.set_xlabel("Score (unique dice count)")
    ax2.set_ylabel("Games")
    ax2.set_title("After training")
    ax2.set_xticks(range(1, num_dice + 1))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
