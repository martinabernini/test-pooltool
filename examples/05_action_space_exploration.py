"""
Example 05 — Action space exploration & shot parameterisation.

Shows how to:
  - discretise phi into N directions for a discrete action space
  - aim helpers (at_ball, at_pocket)
  - apply English (side spin) via 'a' parameter
  - sample a bank shot with elevated theta
"""

from __future__ import annotations

import numpy as np
import pooltool as pt


def discretise_phi(n_angles: int = 36) -> list[float]:
    """Divide 360° into n_angles evenly-spaced directions."""
    return [i * (360.0 / n_angles) for i in range(n_angles)]


def aim_and_shoot(
    system: pt.System,
    target_id: str,
    V0: float = 6.0,
    theta: float = 0.0,
    a: float = 0.0,
    b: float = 0.0,
) -> pt.System:
    """Helper used inside an RL loop: set aim, simulate, return result."""
    s = system.copy()
    phi = pt.aim.at_ball(s, target_id)
    s.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
    pt.simulate(s, inplace=True)
    return s


def main():
    # --- Discrete action demo ---
    angles = discretise_phi(36)
    print(f"Discrete phi values ({len(angles)}): {angles[:6]} ...")

    # --- Aim helpers ---
    table = pt.Table.default()
    balls = pt.get_rack(pt.GameType.NINEBALL, table)
    cue   = pt.Cue(cue_ball_id="cue")
    system = pt.System(table=table, balls=balls, cue=cue)

    phi_to_1 = pt.aim.at_ball(system, "1")
    print(f"\nPhi to ball '1' : {phi_to_1:.2f}°")

    # --- English (side spin) ---
    s_right_english = aim_and_shoot(system, "1", V0=5.0, a=0.3)
    print(f"Right english shot — cue final pos: {s_right_english.balls['cue'].xyz}")

    s_left_english = aim_and_shoot(system, "1", V0=5.0, a=-0.3)
    print(f"Left  english shot — cue final pos: {s_left_english.balls['cue'].xyz}")

    # --- Elevated (masse) shot ---
    s_masse = aim_and_shoot(system, "1", V0=4.0, theta=30.0, b=0.4)
    print(f"Masse shot — cue final pos: {s_masse.balls['cue'].xyz}")

    # --- Sweep V0 and measure energy after shot ---
    print("\n=== V0 sweep ===")
    for v0 in np.linspace(1.0, 10.0, 5):
        s = aim_and_shoot(system, "1", V0=float(v0))
        energy = s.get_system_energy()
        print(f"  V0={v0:.1f}  residual_energy={energy:.4f} J")


if __name__ == "__main__":
    main()
