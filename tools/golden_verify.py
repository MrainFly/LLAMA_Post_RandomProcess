#!/usr/bin/env python3
import argparse
import math
import subprocess
import sys
from dataclasses import dataclass

RNG_MASK = 0x7FFFFFFF
MAX_HISTORY = 256
EPSILON = 1e-8
OFFLINE_TOKEN_IDS = [128000, 128001, 128002, 128003, 128004, 128005, 128006, 128007]
OFFLINE_LOGITS = [
    [2.1, 1.6, 0.5, -0.4, -1.0, -2.0, 0.9, 1.3],
    [1.7, 1.3, 0.4, -0.2, -1.4, -1.9, 1.2, 0.8],
    [1.9, 1.5, 0.6, -0.3, -1.2, -2.1, 0.7, 1.1],
    [2.0, 1.4, 0.2, -0.6, -1.5, -2.0, 0.8, 1.0],
    [1.8, 1.1, 0.3, -0.1, -1.1, -2.3, 1.4, 0.9],
    [2.2, 1.2, 0.7, -0.5, -1.0, -1.8, 0.6, 1.5],
]


@dataclass
class SamplingConfig:
    seed: int = 0x12345678
    temperature: float = 0.8
    top_k: int = 5
    top_p: float = 0.9
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.05
    history_window: int = 64


def next_state(current: int) -> int:
    return (1103515245 * current + 12345) & RNG_MASK


def next_uniform(state: int) -> tuple[float, int]:
    state = next_state(state)
    return state / (RNG_MASK + 1), state


def softmax_kept(candidates: list[dict]) -> bool:
    kept = [c for c in candidates if c["keep"]]
    if not kept:
        return False
    max_logit = max(c["logit"] for c in kept)
    probs = [math.exp(c["logit"] - max_logit) for c in kept]
    total = sum(probs)
    if total <= 0 or not math.isfinite(total):
        return False
    idx = 0
    for c in candidates:
        if not c["keep"]:
            c["prob"] = 0.0
            continue
        c["prob"] = probs[idx] / total
        idx += 1
    return True


def generate_expected(count: int, config: SamplingConfig) -> list[int]:
    rng = config.seed
    history: list[int] = []
    out: list[int] = []

    for step in range(count):
        logits = OFFLINE_LOGITS[step % len(OFFLINE_LOGITS)]
        candidates = []
        effective_history_window = (
            MAX_HISTORY if config.history_window == 0 else config.history_window
        )
        window = max(1, min(effective_history_window, MAX_HISTORY))
        recent_history = history[-window:]

        for token_id, base_logit in zip(OFFLINE_TOKEN_IDS, logits):
            repeat_count = recent_history.count(token_id)
            logit = base_logit
            if repeat_count > 0:
                if config.repeat_penalty > 0.0 and config.repeat_penalty != 1.0:
                    if logit >= 0.0:
                        logit /= config.repeat_penalty
                    else:
                        # Match llama-style behavior for negative logits.
                        logit *= config.repeat_penalty
                logit -= config.presence_penalty
                logit -= repeat_count * config.frequency_penalty
            if config.temperature > 0.0:
                logit /= config.temperature
            candidates.append(
                {"token_id": token_id, "logit": logit, "prob": 0.0, "keep": True}
            )

        if 0 < config.top_k < len(candidates):
            sorted_idx = sorted(
                range(len(candidates)), key=lambda i: candidates[i]["logit"], reverse=True
            )
            for idx in sorted_idx[config.top_k :]:
                candidates[idx]["keep"] = False

        if config.temperature <= 0.0:
            kept = [c for c in candidates if c["keep"]]
            chosen = max(kept, key=lambda c: c["logit"])
            token = chosen["token_id"]
        else:
            if not softmax_kept(candidates):
                kept = [c for c in candidates if c["keep"]]
                token = max(kept, key=lambda c: c["logit"])["token_id"]
            else:
                effective_top_p = max(config.top_p, 1e-6)
                if effective_top_p < 1.0:
                    sorted_kept = sorted(
                        [c for c in candidates if c["keep"]],
                        key=lambda c: c["prob"],
                        reverse=True,
                    )
                    cumulative = 0.0
                    keep_ids: set[int] = set()
                    for idx, c in enumerate(sorted_kept):
                        cumulative += c["prob"]
                        keep_ids.add(c["token_id"])
                        if cumulative >= effective_top_p and idx + 1 < len(sorted_kept):
                            break
                    for c in candidates:
                        if c["keep"] and c["token_id"] not in keep_ids:
                            c["keep"] = False

                if not softmax_kept(candidates):
                    kept = [c for c in candidates if c["keep"]]
                    token = max(kept, key=lambda c: c["logit"])["token_id"]
                else:
                    if config.min_p > 0.0:
                        kept = [c for c in candidates if c["keep"]]
                        max_prob = max((c["prob"] for c in kept), default=0.0)
                        threshold = config.min_p * max_prob
                        for c in candidates:
                            if c["keep"] and c["prob"] + EPSILON < threshold:
                                c["keep"] = False
                        if not any(c["keep"] for c in candidates):
                            max(kept, key=lambda c: c["prob"])["keep"] = True

                    if not softmax_kept(candidates):
                        kept = [c for c in candidates if c["keep"]]
                        token = max(kept, key=lambda c: c["logit"])["token_id"]
                    else:
                        threshold, rng = next_uniform(rng)
                        cumulative = 0.0
                        token = 0
                        for c in candidates:
                            if not c["keep"]:
                                continue
                            cumulative += c["prob"]
                            token = c["token_id"]
                            if threshold <= cumulative:
                                break

        out.append(token)
        history.append(token)
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable")
    parser.add_argument("count", nargs="?", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0x12345678)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--repeat-penalty", type=float, default=1.1)
    parser.add_argument("--frequency-penalty", type=float, default=0.1)
    parser.add_argument("--presence-penalty", type=float, default=0.05)
    parser.add_argument("--history-window", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        print("count must be a positive integer")
        return 2

    config = SamplingConfig(
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repeat_penalty=args.repeat_penalty,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        history_window=args.history_window,
    )

    cmd = [
        args.executable,
        str(args.count),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--min-p",
        str(args.min_p),
        "--repeat-penalty",
        str(args.repeat_penalty),
        "--frequency-penalty",
        str(args.frequency_penalty),
        "--presence-penalty",
        str(args.presence_penalty),
        "--history-window",
        str(args.history_window),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr.strip() or f"C program exited with status {result.returncode}")
        return result.returncode

    c_tokens = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            c_tokens.append(int(stripped))
        except ValueError:
            print(f"Golden verify failed: non-integer output line from C program: {stripped}")
            return 1

    expected = generate_expected(args.count, config)
    if c_tokens != expected:
        print("Golden verify failed")
        print(f"Expected: {expected}")
        print(f"Actual  : {c_tokens}")
        return 1

    print("Golden verify passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
