#!/usr/bin/env python3
"""
EE 5393 HW2
Stochastic simulations for Problems 1 and 2.

Problem 1:
    Gillespie-style SSA for the layered Fibonacci CRN.

Problem 2:
    Gillespie-style SSA for the biquad filter CRN.
    Each 1/8 branch is implemented as three cascaded halving
    reactions (2Z -> Z', 2Z' -> Z'', 2Z'' -> output).
    Each cycle is split into two SSA sub-steps.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from math import comb
from statistics import mean, pstdev
from typing import DefaultDict, Dict, List, Sequence, Tuple

# ============================================================
# General-purpose Gillespie SSA
# ============================================================

# A reaction is (reactants, products, rate) where reactants and
# products are dicts mapping species name -> stoichiometric count.
Reaction = Tuple[Dict[str, int], Dict[str, int], float]


def parse_reaction(reactants_str: str, products_str: str, rate: float) -> Reaction:
    """
    Build a reaction tuple from Aleae-style strings.

    Example:
        parse_reaction("B1 1", "B1a 1 B1y 1 B1d 1", 1.0)
    Returns:
        ({"B1": 1}, {"B1a": 1, "B1y": 1, "B1d": 1}, 1.0)
    """
    def _parse(s: str) -> Dict[str, int]:
        d: Dict[str, int] = {}
        if not s.strip():
            return d
        tokens = s.strip().split()
        for i in range(0, len(tokens), 2):
            d[tokens[i]] = int(tokens[i + 1])
        return d

    return (_parse(reactants_str), _parse(products_str), rate)


def gillespie_ssa(
    reactions: Sequence[Reaction],
    initial_counts: Dict[str, int],
    rng: random.Random | None = None,
    max_events: int = 1_000_000,
) -> Tuple[Dict[str, int], float, int]:
    """
    Run the Gillespie SSA until no reactions can fire.

    Returns (final_counts, total_time, total_events).
    """
    if rng is None:
        rng = random.Random()

    counts: DefaultDict[str, int] = defaultdict(int)
    counts.update(initial_counts)

    t = 0.0
    fired = 0

    while fired < max_events:
        # Compute propensities
        propensities: List[float] = []
        for reactants, _, rate in reactions:
            prop = rate
            possible = True
            for species, needed in reactants.items():
                have = counts[species]
                if have < needed:
                    possible = False
                    break
                prop *= comb(have, needed)
            propensities.append(prop if possible else 0.0)

        total_prop = sum(propensities)
        if total_prop <= 0:
            break

        # Time step
        t += rng.expovariate(total_prop)

        # Choose reaction
        r = rng.random() * total_prop
        cumulative = 0.0
        chosen = len(reactions) - 1
        for i, p in enumerate(propensities):
            cumulative += p
            if cumulative >= r:
                chosen = i
                break

        # Fire reaction
        reactants, products, _ = reactions[chosen]
        for species, n in reactants.items():
            counts[species] -= n
        for species, n in products.items():
            counts[species] += n

        fired += 1

    return dict(counts), t, fired


# ============================================================
# Problem 1: Fibonacci CRN
# ============================================================
#
# Layered reactions for step i:
#     A_i  ->  B_{i+1}                (transfer)
#     B_i  ->  A_{i+1} + B_{i+1}     (fanout)

def fibonacci_ssa(
    a0: int,
    b0: int,
    steps: int = 12,
    seed: int | None = None,
) -> Tuple[List[Tuple[int, int, int, float, int]], float, int]:
    """
    Run one Fibonacci SSA trial layer by layer.

    Returns (history, final_time, total_events) where history is a
    list of (step, A_i, B_i, time, cumulative_events).
    """
    rng = random.Random(seed)

    counts: DefaultDict[str, int] = defaultdict(int)
    counts["A1"] = a0
    counts["B1"] = b0

    history = [(1, a0, b0, 0.0, 0)]
    t = 0.0
    fired = 0

    for i in range(1, steps):
        # Two reactions for this layer
        layer_rxns: List[Reaction] = [
            ({f"A{i}": 1}, {f"B{i+1}": 1}, 1.0),
            ({f"B{i}": 1}, {f"A{i+1}": 1, f"B{i+1}": 1}, 1.0),
        ]

        while counts[f"A{i}"] > 0 or counts[f"B{i}"] > 0:
            a_cnt = counts[f"A{i}"]
            b_cnt = counts[f"B{i}"]
            total_prop = float(a_cnt + b_cnt)
            if total_prop <= 0:
                break

            t += rng.expovariate(total_prop)

            if rng.random() * total_prop < a_cnt:
                counts[f"A{i}"] -= 1
                counts[f"B{i+1}"] += 1
            else:
                counts[f"B{i}"] -= 1
                counts[f"A{i+1}"] += 1
                counts[f"B{i+1}"] += 1

            fired += 1

        history.append((
            i + 1,
            counts[f"A{i+1}"],
            counts[f"B{i+1}"],
            t,
            fired,
        ))

    return history, t, fired


def summarize_fibonacci_trials(
    a0: int, b0: int, trials: int = 1000, steps: int = 12,
) -> dict:
    times = []
    for seed in range(1, trials + 1):
        _, t, _ = fibonacci_ssa(a0, b0, steps=steps, seed=seed)
        times.append(t)

    hist0, _, _ = fibonacci_ssa(a0, b0, steps=steps, seed=1)
    return {
        "final_A": hist0[-1][1],
        "final_B": hist0[-1][2],
        "time_mean": mean(times),
        "time_std": pstdev(times),
        "time_min": min(times),
        "time_max": max(times),
    }


# ============================================================
# Problem 2: Biquad filter CRN
# ============================================================
#
# State equations (Direct Form II, all coefficients 1/8):
#     A_n  = X_n + S1_n/8 + S2_n/8
#     Y_n  = A_n/8 + S1_n/8 + S2_n/8
#     S1_{n+1} = A_n
#     S2_{n+1} = S1_n
#
# CRN implementation: two Gillespie sub-steps per cycle.
#
# Sub-step 1: accumulate A = X + B1/8 + B2/8, save copies
# Sub-step 2: fan out A, compute Y = A/8 + B1/8 + B2/8

BIQUAD_SUBSTEP1 = [
    parse_reaction("X 1",      "A 1",                  1.0),
    parse_reaction("B1 1",     "B1a 1 B1y 1 B1d 1",   1.0),
    parse_reaction("B2 1",     "B2a 1 B2y 1",          1.0),
    # B1/8 -> A  (cascaded halving)
    parse_reaction("B1a 2",    "B1a2 1",               1.0),
    parse_reaction("B1a2 2",   "B1a3 1",               1.0),
    parse_reaction("B1a3 2",   "A 1",                  1.0),
    # B2/8 -> A  (cascaded halving)
    parse_reaction("B2a 2",    "B2a2 1",               1.0),
    parse_reaction("B2a2 2",   "B2a3 1",               1.0),
    parse_reaction("B2a3 2",   "A 1",                  1.0),
    # full B1 -> D2
    parse_reaction("B1d 1",    "D2 1",                 1.0),
]

BIQUAD_SUBSTEP2 = [
    parse_reaction("A 1",      "Ay 1 D1 1",            1.0),
    # A/8 -> Y  (cascaded halving)
    parse_reaction("Ay 2",     "Ay2 1",                1.0),
    parse_reaction("Ay2 2",    "Ay3 1",                1.0),
    parse_reaction("Ay3 2",    "Y 1",                  1.0),
    # B1/8 -> Y
    parse_reaction("B1y 2",    "B1y2 1",               1.0),
    parse_reaction("B1y2 2",   "B1y3 1",               1.0),
    parse_reaction("B1y3 2",   "Y 1",                  1.0),
    # B2/8 -> Y
    parse_reaction("B2y 2",    "B2y2 1",               1.0),
    parse_reaction("B2y2 2",   "B2y3 1",               1.0),
    parse_reaction("B2y3 2",   "Y 1",                  1.0),
]


BiquadRow = Tuple[int, int, int, int, int, int, int]


def biquad_crn_cycle(
    X_val: int, B1_val: int, B2_val: int,
    rng: random.Random,
) -> Tuple[int, int, int, float, int]:
    """
    Run one biquad RGB cycle as two Gillespie sub-steps.

    Returns (Y, D1, D2, total_time, total_events).
    """
    # Sub-step 1: accumulate A
    init1: Dict[str, int] = {
        "X": X_val, "B1": B1_val, "B2": B2_val,
        "A": 0, "B1a": 0, "B1y": 0, "B1d": 0,
        "B2a": 0, "B2y": 0,
        "B1a2": 0, "B1a3": 0,
        "B2a2": 0, "B2a3": 0,
        "D2": 0,
    }
    r1, t1, e1 = gillespie_ssa(BIQUAD_SUBSTEP1, init1, rng)

    # Sub-step 2: fan out A, compute Y
    init2: Dict[str, int] = {
        "A": r1.get("A", 0),
        "Ay": 0, "D1": 0,
        "Ay2": 0, "Ay3": 0, "Y": 0,
        "B1y": r1.get("B1y", 0),
        "B1y2": 0, "B1y3": 0,
        "B2y": r1.get("B2y", 0),
        "B2y2": 0, "B2y3": 0,
    }
    r2, t2, e2 = gillespie_ssa(BIQUAD_SUBSTEP2, init2, rng)

    return (
        r2.get("Y", 0),
        r2.get("D1", 0),
        r1.get("D2", 0),
        t1 + t2,
        e1 + e2,
    )


def biquad_crn_simulate(
    inputs: Sequence[int],
    seed: int | None = None,
) -> List[BiquadRow]:
    """
    Simulate the biquad CRN for a sequence of inputs.

    Returns list of (cycle, X, S1, S2, Y, D1_new, D2_new).
    """
    rng = random.Random(seed)
    S1, S2 = 0, 0
    rows: List[BiquadRow] = []

    for n, X in enumerate(inputs, start=1):
        Y, D1_new, D2_new, _, _ = biquad_crn_cycle(X, S1, S2, rng)
        rows.append((n, X, S1, S2, Y, D1_new, D2_new))
        S1, S2 = D1_new, D2_new

    return rows


def deterministic_biquad(inputs: Sequence[int]) -> List[Tuple[int, float, float]]:
    """
    Deterministic reference:
        A_n = X_n + S1/8 + S2/8
        Y_n = A_n/8 + S1/8 + S2/8
    Returns list of (cycle, A_exact, Y_exact).
    """
    s1, s2 = 0.0, 0.0
    out = []
    for n, x in enumerate(inputs, start=1):
        a = x + s1 / 8.0 + s2 / 8.0
        y = a / 8.0 + s1 / 8.0 + s2 / 8.0
        out.append((n, a, y))
        s2, s1 = s1, a
    return out


def summarize_biquad_trials(
    inputs: Sequence[int], trials: int = 1000,
) -> List[dict]:
    ys: List[List[int]] = [[] for _ in inputs]

    for seed in range(1, trials + 1):
        rows = biquad_crn_simulate(inputs, seed=seed)
        for i, row in enumerate(rows):
            ys[i].append(row[4])  # Y value

    out = []
    for i, vals in enumerate(ys, start=1):
        out.append({
            "cycle": i,
            "mean": mean(vals),
            "std": pstdev(vals),
            "min": min(vals),
            "max": max(vals),
        })
    return out


# ============================================================
# Printing
# ============================================================

def print_fib_table(history, label: str) -> None:
    print(label)
    print(f"{'Step':>4} {'A':>8} {'B':>8} {'time':>12} {'events':>8}")
    print("-" * 48)
    for step, a, b, t, fired in history:
        print(f"{step:>4} {a:>8} {b:>8} {t:>12.6f} {fired:>8}")
    print()


def print_reactions(label: str, rxns: Sequence[Reaction]) -> None:
    print(label)
    for reactants, products, rate in rxns:
        r_str = " + ".join(
            f"{n}{s}" if n > 1 else s for s, n in reactants.items()
        )
        p_str = " + ".join(
            f"{n}{s}" if n > 1 else s for s, n in products.items()
        )
        print(f"    {r_str:20s} -> {p_str}")
    print()


def print_biquad_table(rows: Sequence[BiquadRow]) -> None:
    print(f"{'cyc':>3} {'X':>5} {'S1':>6} {'S2':>6} {'Y':>5} {'D1':>6} {'D2':>6}")
    print("-" * 44)
    for cycle, X, S1, S2, Y, D1, D2 in rows:
        print(f"{cycle:>3} {X:>5} {S1:>6} {S2:>6} {Y:>5} {D1:>6} {D2:>6}")
    print()


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5393, help="base seed")
    parser.add_argument("--trials", type=int, default=1000, help="number of trials")
    args = parser.parse_args()

    # ── Problem 1 ──
    print("=" * 60)
    print("  Problem 1: Fibonacci via CRN (Gillespie SSA)")
    print("=" * 60)

    print("\nMolecular reactions (per layer i = 1..11):")
    print("    A_i  ->  B_{i+1}              (transfer)")
    print("    B_i  ->  A_{i+1} + B_{i+1}    (fanout)")
    print()

    hist01, _, _ = fibonacci_ssa(0, 1, seed=args.seed)
    hist37, _, _ = fibonacci_ssa(3, 7, seed=args.seed + 1)

    print_fib_table(hist01, "Sample run: initial state (0, 1)")
    print_fib_table(hist37, "Sample run: initial state (3, 7)")

    stats01 = summarize_fibonacci_trials(0, 1, trials=args.trials)
    stats37 = summarize_fibonacci_trials(3, 7, trials=args.trials)

    print("Multi-trial summary (counts deterministic; timing stochastic)")
    print(
        f"  (0,1): (A12,B12)=({stats01['final_A']},{stats01['final_B']}), "
        f"mean time={stats01['time_mean']:.4f}, "
        f"std={stats01['time_std']:.4f}"
    )
    print(
        f"  (3,7): (A12,B12)=({stats37['final_A']},{stats37['final_B']}), "
        f"mean time={stats37['time_mean']:.4f}, "
        f"std={stats37['time_std']:.4f}"
    )

    # ── Problem 2 ──
    print("\n" + "=" * 60)
    print("  Problem 2: Biquad filter via CRN (Gillespie SSA)")
    print("=" * 60)

    print("\nState equations:")
    print("    A_n  = X_n + S1_n/8 + S2_n/8")
    print("    Y_n  = A_n/8 + S1_n/8 + S2_n/8")
    print("    S1_{n+1} = A_n     S2_{n+1} = S1_n")

    print_reactions(
        "\nSub-step 1 reactions (accumulate A = X + B1/8 + B2/8):",
        BIQUAD_SUBSTEP1,
    )
    print_reactions(
        "Sub-step 2 reactions (compute Y = A/8 + B1/8 + B2/8, store D1):",
        BIQUAD_SUBSTEP2,
    )

    inputs = [100, 5, 500, 20, 250]
    print(f"Input sequence: X = {inputs}\n")

    # Single sample run
    rows = biquad_crn_simulate(inputs, seed=args.seed)
    print("Sample CRN run:")
    print_biquad_table(rows)
    print(f"Final state: S1 = {rows[-1][5]}, S2 = {rows[-1][6]}\n")

    # Multi-trial summary
    summary = summarize_biquad_trials(inputs, trials=args.trials)
    det = deterministic_biquad(inputs)

    print("Multi-trial summary vs deterministic reference:")
    print(f"{'cyc':>3} {'E[Y]':>10} {'std':>8} {'min':>5} {'max':>5} {'det Y':>12} {'det A':>12}")
    print("-" * 62)
    for row, det_row in zip(summary, det):
        print(
            f"{row['cycle']:>3} {row['mean']:>10.3f} {row['std']:>8.3f} "
            f"{row['min']:>5} {row['max']:>5} "
            f"{det_row[2]:>12.6f} {det_row[1]:>12.4f}"
        )
    print()


if __name__ == "__main__":
    main()
