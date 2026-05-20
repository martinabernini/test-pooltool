"""
Example 03 — Reward function candidates for RL training.

Three reward strategies:
  A. Sparse   — +1 per ball pocketed, -10 scratch
  B. Dense    — continuous reward from cue-ball proximity to target
  C. Combined — sparse + shaping term
"""

from __future__ import annotations

import numpy as np
import pooltool as pt


def _pocketed_ids(system: pt.System) -> set[str]:
    return {
        bid
        for bid, ball in system.balls.items()
        if ball.state.s == pt.constants.pocketed
    }


def _scratch(system: pt.System) -> bool:
    """Cue ball pocketed = scratch."""
    cue = system.balls.get("cue")
    return cue is not None and cue.state.s == pt.constants.pocketed


# --- A. Sparse reward ---

def reward_sparse(
    system_before: pt.System,
    system_after: pt.System,
) -> float:
    pocketed_before = _pocketed_ids(system_before)
    pocketed_after = _pocketed_ids(system_after)
    new_pocketed = pocketed_after - pocketed_before - {"cue"}

    reward = float(len(new_pocketed))
    if _scratch(system_after):
        reward -= 10.0
    return reward


# --- B. Dense proximity reward ---

def reward_dense(system_after: pt.System, target_ball_id: str = "1") -> float:
    """
    Negative distance from cue ball to target ball after shot settles.
    Encourages the agent to move cue into good position.
    """
    if target_ball_id not in system_after.balls:
        return 0.0
    cue_xyz = system_after.balls["cue"].xyz
    tgt_xyz = system_after.balls[target_ball_id].xyz
    dist = float(np.linalg.norm(np.array(cue_xyz[:2]) - np.array(tgt_xyz[:2])))
    return -dist  # maximise => minimise distance


# --- C. Combined reward ---

def reward_combined(
    system_before: pt.System,
    system_after: pt.System,
    target_ball_id: str = "1",
    w_sparse: float = 1.0,
    w_dense: float = 0.1,
) -> float:
    return (
        w_sparse * reward_sparse(system_before, system_after)
        + w_dense * reward_dense(system_after, target_ball_id)
    )


# --- Demo ---

def main():
    table = pt.Table.default()
    balls = pt.get_rack(pt.GameType.NINEBALL, table)
    cue = pt.Cue(cue_ball_id="cue")
    system = pt.System(table=table, balls=balls, cue=cue)
    system_before = system.copy()

    system.cue.set_state(V0=9.0, phi=pt.aim.at_ball(system, "1"))
    pt.simulate(system, inplace=True)

    print(f"Sparse   reward: {reward_sparse(system_before, system):.2f}")
    print(f"Dense    reward: {reward_dense(system):.4f}")
    print(f"Combined reward: {reward_combined(system_before, system):.4f}")


if __name__ == "__main__":
    main()
