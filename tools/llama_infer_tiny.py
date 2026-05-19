#!/usr/bin/env python3
"""Tiny llama-like inference demo (no external dependencies)."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass


TOKEN_IDS = [128000, 128001, 128002, 128003, 128004, 128005, 128006, 128007]
TOKEN_INDEX = {token: i for i, token in enumerate(TOKEN_IDS)}
DIM = 8
RMS_NORM_EPSILON = 1e-6
MIN_SUM_THRESHOLD = 1e-10


def _sinusoidal_weight_init(seed: int, i: int, j: int, scale: float) -> float:
    return math.sin(seed * 0.01 + i * 0.37 + j * 0.13) * scale


EMBED = [
    [_sinusoidal_weight_init(11, i, j, 0.6) for j in range(DIM)]
    for i in range(len(TOKEN_IDS))
]
W_HIDDEN = [[_sinusoidal_weight_init(17, i, j, 0.3) for j in range(DIM)] for i in range(DIM)]
W_OUT = [[_sinusoidal_weight_init(23, i, j, 0.5) for j in range(DIM)] for i in range(len(TOKEN_IDS))]


@dataclass
class TinyState:
    hidden: list[float]


def matvec(matrix: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(row[k] * vec[k] for k in range(len(vec))) for row in matrix]


def rmsnorm(vec: list[float], eps: float = RMS_NORM_EPSILON) -> list[float]:
    mean_sq = sum(v * v for v in vec) / len(vec)
    inv = 1.0 / math.sqrt(mean_sq + eps)
    return [v * inv for v in vec]


def inference_step(state: TinyState, token_id: int) -> list[float]:
    idx = TOKEN_INDEX.get(token_id, 0)
    emb = EMBED[idx]
    recurrent_output = matvec(W_HIDDEN, state.hidden)
    mixed = [math.tanh(0.7 * recurrent_output[i] + emb[i]) for i in range(DIM)]
    state.hidden = rmsnorm(mixed)
    return matvec(W_OUT, state.hidden)


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    if s < MIN_SUM_THRESHOLD:
        return [1.0 / len(logits)] * len(logits)
    return [v / s for v in exps]


def argmax(values: list[float]) -> int:
    best_idx = 0
    best = values[0]
    for i, v in enumerate(values[1:], start=1):
        if v > best:
            best = v
            best_idx = i
    return best_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiny llama-like inference demo that outputs logits and greedy tokens."
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="*",
        default=[128000, 128001, 128002],
        help="prompt tokens (default: 128000 128001 128002)",
    )
    parser.add_argument("--steps", type=int, default=6, help="generation steps")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        print("Error: steps must be > 0", file=sys.stderr)
        return 2
    if len(args.tokens) == 0:
        print("Error: --tokens cannot be empty", file=sys.stderr)
        return 2

    state = TinyState(hidden=[0.0] * DIM)
    prompt = args.tokens

    for token in prompt:
        _ = inference_step(state, token)

    cur = prompt[-1]
    print("# tiny inference result")
    for step in range(args.steps):
        logits = inference_step(state, cur)
        probs = softmax(logits)
        next_idx = argmax(probs)
        cur = TOKEN_IDS[next_idx]
        print(
            f"step={step:02d} token={cur} prob={probs[next_idx]:.6f} "
            f"logits={[round(x, 4) for x in logits]}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
