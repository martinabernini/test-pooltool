"""
Example 06 — Collect & save trajectories for offline RL / imitation learning.

Generates N random shots, stores (obs_before, action, reward, obs_after)
as a numpy .npz file — ready to feed into a replay buffer or BC dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pooltool as pt

BALL_ORDER = ["cue", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
OBS_DIM    = len(BALL_ORDER) * 5
ACT_DIM    = 4  # [V0, phi, theta, b]


def _extract_obs(system: pt.System) -> np.ndarray:
    obs = []
    for bid in BALL_ORDER:
        if bid not in system.balls:
            obs.extend([0.0, 0.0, 0.0, 0.0, 1.0])
            continue
        ball = system.balls[bid]
        x, y, _   = ball.xyz
        vx, vy, _ = ball.vel
        pocketed   = float(ball.state.s == pt.constants.pocketed)
        obs.extend([x, y, vx, vy, pocketed])
    return np.array(obs, dtype=np.float32)


def _pocketed_count(system: pt.System) -> int:
    return sum(
        1
        for bid, ball in system.balls.items()
        if bid != "cue" and ball.state.s == pt.constants.pocketed
    )


def collect(n_shots: int = 1000, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obses, actions, rewards, next_obses, dones = [], [], [], [], []

    for i in range(n_shots):
        table  = pt.Table.default()
        balls  = pt.get_rack(pt.GameType.NINEBALL, table)
        cue    = pt.Cue(cue_ball_id="cue")
        system = pt.System(table=table, balls=balls, cue=cue)

        obs_before = _extract_obs(system)

        # Random action
        V0    = float(rng.uniform(0.5, 10.0))
        phi   = float(rng.uniform(0.0, 360.0))
        theta = float(rng.uniform(0.0, 20.0))
        b     = float(rng.uniform(-0.5, 0.5))
        action = np.array([V0, phi, theta, b], dtype=np.float32)

        system.cue.set_state(V0=V0, phi=phi, theta=theta, a=0.0, b=b)
        pt.simulate(system, inplace=True)

        obs_after = _extract_obs(system)
        reward    = float(_pocketed_count(system))
        done      = _pocketed_count(system) == len(BALL_ORDER) - 1

        obses.append(obs_before)
        actions.append(action)
        rewards.append(reward)
        next_obses.append(obs_after)
        dones.append(done)

        if (i + 1) % 100 == 0:
            print(f"  collected {i+1}/{n_shots} shots")

    return {
        "obs":      np.stack(obses),
        "action":   np.stack(actions),
        "reward":   np.array(rewards, dtype=np.float32),
        "next_obs": np.stack(next_obses),
        "done":     np.array(dones,   dtype=bool),
    }


def main():
    out_path = Path(__file__).parent / "dataset.npz"
    print(f"Collecting 200 random shots ...")
    data = collect(n_shots=200, seed=0)

    np.savez(out_path, **data)
    print(f"\nSaved to {out_path}")
    for k, v in data.items():
        print(f"  {k}: {v.shape}  dtype={v.dtype}")

    print(f"\nMean reward : {data['reward'].mean():.3f}")
    print(f"Max reward  : {data['reward'].max():.0f}")


if __name__ == "__main__":
    main()
