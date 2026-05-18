#!/usr/bin/env python3
import argparse
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
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


GOLDEN_BY_CONFIG: dict[SamplingConfig, list[int]] = {
    SamplingConfig(): [128000, 128007, 128001, 128000, 128006, 128002, 128000, 128006, 128000, 128000, 128007, 128000, 128001, 128000, 128006, 128001, 128007, 128000, 128007, 128001, 128001, 128002, 128002, 128007, 128000, 128006, 128001, 128007, 128000, 128002, 128001, 128001, 128000, 128007, 128003, 128000, 128002, 128006, 128001, 128006, 128003, 128000, 128000, 128002, 128007, 128007, 128000, 128007, 128000, 128002, 128007, 128001, 128007, 128006, 128001, 128006, 128001, 128000, 128006, 128006, 128006, 128001, 128007, 128002],
    SamplingConfig(top_k=1): [128000, 128000, 128001, 128000, 128006, 128000, 128000, 128001, 128000, 128000, 128006, 128007, 128001, 128006, 128001, 128000, 128006, 128007, 128000, 128001, 128001, 128000, 128006, 128007, 128000, 128006, 128001, 128000, 128006, 128007, 128007, 128001, 128002, 128000, 128006, 128007, 128001, 128006, 128001, 128000, 128006, 128007, 128000, 128002, 128001, 128000, 128006, 128007, 128007, 128002, 128001, 128000, 128006, 128007, 128001, 128002, 128002, 128000, 128006, 128007, 128001, 128002, 128001, 128000],
    SamplingConfig(top_p=1.0): [128000, 128007, 128001, 128000, 128006, 128006, 128000, 128007, 128000, 128000, 128002, 128007, 128000, 128001, 128006, 128001, 128007, 128000, 128007, 128001, 128000, 128002, 128002, 128007, 128000, 128006, 128001, 128006, 128000, 128002, 128000, 128000, 128001, 128002, 128003, 128000, 128002, 128006, 128001, 128007, 128003, 128000, 128001, 128002, 128000, 128006, 128001, 128000, 128007, 128007, 128006, 128001, 128007, 128002, 128001, 128006, 128001, 128000, 128006, 128006, 128007, 128001, 128002, 128002],
    SamplingConfig(top_k=0): [128000, 128007, 128001, 128000, 128006, 128006, 128000, 128007, 128000, 128000, 128002, 128007, 128000, 128001, 128006, 128001, 128007, 128000, 128007, 128001, 128000, 128002, 128003, 128007, 128000, 128006, 128001, 128006, 128000, 128002, 128000, 128007, 128001, 128002, 128003, 128000, 128002, 128006, 128001, 128007, 128003, 128000, 128001, 128003, 128000, 128007, 128000, 128007, 128000, 128007, 128007, 128001, 128002, 128006, 128001, 128006, 128001, 128000, 128006, 128004, 128006, 128001, 128006, 128002],
    SamplingConfig(min_p=0.0): [128000, 128007, 128001, 128000, 128006, 128002, 128000, 128006, 128000, 128000, 128007, 128000, 128001, 128000, 128006, 128001, 128007, 128000, 128007, 128001, 128001, 128002, 128002, 128007, 128000, 128006, 128001, 128007, 128000, 128002, 128001, 128001, 128000, 128007, 128003, 128000, 128002, 128006, 128001, 128006, 128003, 128000, 128000, 128002, 128007, 128007, 128000, 128007, 128000, 128002, 128007, 128001, 128007, 128006, 128001, 128006, 128001, 128000, 128006, 128006, 128006, 128001, 128007, 128002],
    SamplingConfig(repeat_penalty=2.0, frequency_penalty=0.8, presence_penalty=0.5): [128000, 128007, 128001, 128006, 128002, 128002, 128000, 128006, 128003, 128001, 128003, 128007, 128004, 128001, 128005, 128007, 128002, 128000, 128000, 128006, 128001, 128001, 128000, 128007, 128006, 128003, 128002, 128002, 128003, 128006, 128004, 128001, 128007, 128002, 128004, 128000, 128003, 128006, 128007, 128007, 128004, 128000, 128001, 128000, 128002, 128007, 128003, 128006, 128001, 128002, 128000, 128005, 128004, 128001, 128006, 128003, 128005, 128007, 128006, 128007, 128005, 128003, 128001, 128004],
    SamplingConfig(history_window=0): [128000, 128007, 128001, 128000, 128006, 128002, 128000, 128006, 128000, 128000, 128007, 128000, 128001, 128000, 128006, 128001, 128007, 128000, 128007, 128001, 128001, 128002, 128002, 128007, 128000, 128006, 128001, 128007, 128000, 128002, 128001, 128001, 128000, 128007, 128003, 128000, 128002, 128006, 128001, 128006, 128003, 128000, 128000, 128002, 128007, 128007, 128000, 128007, 128000, 128002, 128007, 128001, 128007, 128006, 128001, 128006, 128001, 128000, 128006, 128006, 128006, 128001, 128007, 128002],
}


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
    if config not in GOLDEN_BY_CONFIG:
        print(f"Golden verify failed: unsupported parameter set: {config}")
        return 2

    expected_full = GOLDEN_BY_CONFIG[config]
    if args.count > len(expected_full):
        print(
            f"Golden verify failed: count={args.count} exceeds golden length={len(expected_full)}"
        )
        return 2
    expected = expected_full[: args.count]

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

    if c_tokens != expected:
        print("Golden verify failed")
        print(f"Expected: {expected}")
        print(f"Actual  : {c_tokens}")
        return 1

    print("Golden verify passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
