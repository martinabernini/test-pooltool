"""
Example 07 — Parameter sweep until a ball gets pocketed.

Runs a deterministic sweep over shot parameters, stopping at the first trial
that pockets any object ball. Prints summary statistics for the full
experiment, including pocket rate and which parameters produced the first
successful shot.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pooltool as pt


@dataclass(frozen=True)
class ShotParams:
    V0: float
    phi_offset: float
    theta: float
    a: float
    b: float


@dataclass
class TrialResult:
    params: ShotParams
    pocketed_ids: tuple[str, ...]
    cue_scratched: bool


def _make_system() -> pt.System:
    table = pt.Table.default()
    balls = pt.get_rack(pt.GameType.NINEBALL, table)
    cue = pt.Cue(cue_ball_id="cue")
    return pt.System(table=table, balls=balls, cue=cue)


def _pocketed_ids(system: pt.System) -> list[str]:
    return [
        ball_id
        for ball_id, ball in system.balls.items()
        if ball.state.s == pt.constants.pocketed
    ]


def _cue_scratched(system: pt.System) -> bool:
    cue = system.balls.get("cue")
    return cue is not None and cue.state.s == pt.constants.pocketed


def _run_trial(system: pt.System, params: ShotParams) -> TrialResult:
    shot = system.copy()
    base_phi = pt.aim.at_ball(shot, "1")

    shot.cue.set_state(
        V0=params.V0,
        phi=base_phi + params.phi_offset,
        theta=params.theta,
        a=params.a,
        b=params.b,
    )
    pt.simulate(shot, inplace=True)

    pocketed_ids = tuple(ball_id for ball_id in _pocketed_ids(shot) if ball_id != "cue")
    return TrialResult(
        params=params,
        pocketed_ids=pocketed_ids,
        cue_scratched=_cue_scratched(shot),
    )


def _parameter_grid() -> Iterable[ShotParams]:
    v0_values = np.linspace(2.5, 8.0, 8)
    phi_offsets = np.linspace(-8.0, 8.0, 9)
    theta_values = [0.0, 4.0, 8.0]
    a_values = [-0.25, 0.0, 0.25]
    b_values = [-0.20, 0.0, 0.20]

    for V0, phi_offset, theta, a, b in product(
        v0_values, phi_offsets, theta_values, a_values, b_values
    ):
        yield ShotParams(
            V0=float(V0),
            phi_offset=float(phi_offset),
            theta=float(theta),
            a=float(a),
            b=float(b),
        )


def _summarise(results: list[TrialResult]) -> None:
    attempts = len(results)
    pocketed_trials = [result for result in results if result.pocketed_ids]
    scratched_trials = [result for result in results if result.cue_scratched]
    pocketed_balls = [ball_id for result in results for ball_id in result.pocketed_ids]

    print("\n=== Experiment statistics ===")
    print(f"Trials run           : {attempts}")
    print(f"Trials with pocket    : {len(pocketed_trials)}")
    print(f"Pocket rate           : {len(pocketed_trials) / attempts:.2%}")
    print(f"Trials with scratch   : {len(scratched_trials)}")
    print(f"Scratch rate          : {len(scratched_trials) / attempts:.2%}")

    if pocketed_balls:
        unique, counts = np.unique(pocketed_balls, return_counts=True)
        order = np.argsort(counts)[::-1]
        print("Most pocketed balls   :")
        for idx in order[:5]:
            print(f"  {unique[idx]} -> {counts[idx]} times")
    else:
        print("Most pocketed balls   : none")


def main() -> None:
    system = _make_system()
    results: list[TrialResult] = []
    first_success: TrialResult | None = None

    for trial_index, params in enumerate(_parameter_grid(), start=1):
        result = _run_trial(system, params)
        results.append(result)

        pocketed = ", ".join(result.pocketed_ids) if result.pocketed_ids else "none"
        print(
            f"Trial {trial_index:03d} | V0={params.V0:4.1f}  "
            f"dPhi={params.phi_offset:5.1f}  theta={params.theta:4.1f}  "
            f"a={params.a:+.2f}  b={params.b:+.2f}  pocketed={pocketed}"
        )

        if result.pocketed_ids:
            first_success = result
            break

    if first_success is None:
        print("\nNo pocketed ball was found in the parameter sweep.")
    else:
        params = first_success.params
        print("\n=== First pocketed shot ===")
        print(f"Ball(s) pocketed : {', '.join(first_success.pocketed_ids)}")
        print(
            f"Parameters       : V0={params.V0:.2f}, dPhi={params.phi_offset:.2f}, "
            f"theta={params.theta:.2f}, a={params.a:.2f}, b={params.b:.2f}"
        )

    _summarise(results)


if __name__ == "__main__":
    main()