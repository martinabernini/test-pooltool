"""
Example 01 — Basic shot simulation.
Rack nine-ball, aim cue at the 1-ball, simulate, print final ball positions.
"""

import pooltool as pt


def main():
    table = pt.Table.default()
    balls = pt.get_rack(pt.GameType.NINEBALL, table)
    cue = pt.Cue(cue_ball_id="cue")

    shot = pt.System(table=table, balls=balls, cue=cue)

    # Aim at ball "1", hit at 8 m/s, no spin
    shot.cue.set_state(
        V0=8.0,
        phi=pt.aim.at_ball(shot, "1"),
        theta=0.0,
        a=0.0,
        b=0.0,
    )

    pt.simulate(shot, inplace=True)

    print("=== Final ball positions ===")
    for ball_id, ball in shot.balls.items():
        x, y, z = ball.xyz
        print(f"  {ball_id:>4s}  x={x:.3f}  y={y:.3f}  pocketed={ball.state.s == pt.BallState.POCKETED}")

    print(f"\nTotal events : {len(shot.events)}")
    print(f"Sim time (s) : {shot.t:.4f}")

    # Visualize (comment out in headless training)
    # pt.ShotViewer().show(shot)


if __name__ == "__main__":
    main()
