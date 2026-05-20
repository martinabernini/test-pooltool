"""
Example 02 — Extract an observation vector from a System.
Shows how to build the state space an RL agent will consume.

Observation per ball (6 floats): [x, y, vx, vy, is_pocketed, ball_index]
Cue action (4 floats): [V0, phi, theta, b]
"""

from __future__ import annotations

import numpy as np
import pooltool as pt


BALL_ORDER = ["cue", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def extract_observation(system: pt.System) -> np.ndarray:
    """
    Returns flat float32 array of shape (len(BALL_ORDER) * 5,).
    Per ball: [x, y, vx, vy, is_pocketed]
    """
    obs = []
    for bid in BALL_ORDER:
        if bid not in system.balls:
            obs.extend([0.0, 0.0, 0.0, 0.0, 1.0])
            continue
        ball = system.balls[bid]
        x, y, _ = ball.xyz
        vx, vy, _ = ball.vel
        pocketed = float(ball.state.s == pt.constants.BallState.POCKETED)
        obs.extend([x, y, vx, vy, pocketed])
    return np.array(obs, dtype=np.float32)


def extract_events_summary(system: pt.System) -> dict:
    """Count event types — useful for shaping rewards."""
    counts: dict[str, int] = {}
    for event in system.events:
        key = event.event_type.name
        counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    table = pt.Table.default()
    balls = pt.get_rack(pt.GameType.NINEBALL, table)
    cue = pt.Cue(cue_ball_id="cue")
    system = pt.System(table=table, balls=balls, cue=cue)

    print("=== Observation BEFORE shot ===")
    obs_before = extract_observation(system)
    print(f"Shape: {obs_before.shape}")
    print(f"Vector: {obs_before}")

    system.cue.set_state(V0=7.0, phi=pt.aim.at_ball(system, "1"))
    pt.simulate(system, inplace=True)

    print("\n=== Observation AFTER shot ===")
    obs_after = extract_observation(system)
    print(f"Vector: {obs_after}")

    print("\n=== Event counts (for reward shaping) ===")
    for name, count in extract_events_summary(system).items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
