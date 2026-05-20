"""
Example 04 — Gymnasium-compatible RL environment wrapping pooltool.

Action space : Box(4,) = [V0, phi, theta, b]
               V0    in [0.5, 10.0]  m/s
               phi   in [0,   360]   degrees
               theta in [0,    60]   degrees
               b     in [-0.5, 0.5]  top/bottom spin

Observation  : Box(50,) = 10 balls × 5 floats [x, y, vx, vy, pocketed]

Termination  : all non-cue balls pocketed OR max_steps reached
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pooltool as pt

BALL_ORDER = ["cue", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
OBS_DIM = len(BALL_ORDER) * 5


class BilliardEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, game_type: pt.GameType = pt.GameType.NINEBALL, max_steps: int = 20):
        super().__init__()
        self.game_type = game_type
        self.max_steps = max_steps
        self._step = 0
        self._system: pt.System | None = None
            # [V0, phi, theta, b]
        low  = np.array([0.5,   0.0, 0.0, -0.5], dtype=np.float32)
        high = np.array([10.0, 360.0, 60.0,  0.5], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # positions + velocities + pocketed flag, normalised to [0,1] / [-1,1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        table = pt.Table.default()
        balls = pt.get_rack(self.game_type, table)
        cue   = pt.Cue(cue_ball_id="cue")
        self._system = pt.System(table=table, balls=balls, cue=cue)
        self._step = 0
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        assert self._system is not None, "Call reset() first"

        V0, phi, theta, b = action.tolist()
        system_before = self._system.copy()

        self._system.cue.set_state(V0=V0, phi=phi, theta=theta, a=0.0, b=b)
        pt.simulate(self._system, inplace=True)
        self._step += 1

        obs     = self._get_obs()
        reward  = self._compute_reward(system_before)
        terminated = self._all_pocketed()
        truncated  = self._step >= self.max_steps

        info = {
            "step": self._step,
            "pocketed": list(self._pocketed_ids()),
            "scratch": self._scratch(),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        pt.show(self._system)

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = []
        for bid in BALL_ORDER:
            if bid not in self._system.balls:
                obs.extend([0.0, 0.0, 0.0, 0.0, 1.0])
                continue
            ball = self._system.balls[bid]
            x, y, _   = ball.xyz
            vx, vy, _ = ball.vel
            pocketed   = float(ball.state.s == pt.constants.pocketed)
            obs.extend([x, y, vx, vy, pocketed])
        return np.array(obs, dtype=np.float32)

    def _pocketed_ids(self) -> set[str]:
        return {
            bid
            for bid, ball in self._system.balls.items()
            if ball.state.s == pt.constants.pocketed
        }

    def _scratch(self) -> bool:
        cue = self._system.balls.get("cue")
        return cue is not None and cue.state.s == pt.constants.pocketed

    def _all_pocketed(self) -> bool:
        return all(
            self._system.balls[bid].state.s == pt.constants.pocketed
            for bid in BALL_ORDER
            if bid != "cue" and bid in self._system.balls
        )

    def _compute_reward(self, system_before: pt.System) -> float:
        pocketed_before = {
            bid
            for bid, ball in system_before.balls.items()
            if ball.state.s == pt.constants.pocketed and bid != "cue"
        }
        new_pocketed = self._pocketed_ids() - pocketed_before - {"cue"}
        reward = float(len(new_pocketed))
        if self._scratch():
            reward -= 5.0
        return reward


# --- Sanity check with random policy ---

def main():
    env = BilliardEnv()
    obs, _ = env.reset()
    print(f"Obs shape : {obs.shape}")
    print(f"Act shape : {env.action_space.shape}")

    total_reward = 0.0
    for t in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  step={t+1}  reward={reward:.2f}  pocketed={info['pocketed']}  done={terminated or truncated}")
        if terminated or truncated:
            break

    print(f"Total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
