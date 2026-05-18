#!/usr/bin/env python3
import subprocess
import sys

GLOBAL_SEED = 0x12345678
TOKEN_TABLE = [
    128000,
    128001,
    128002,
    128003,
    128004,
    128005,
    128006,
    128007,
    128008,
    128009,
    128010,
    128011,
    128012,
    128013,
    128014,
    128015,
]


def generate_expected(count: int) -> list[int]:
    state = GLOBAL_SEED
    out = []
    for _ in range(count):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        out.append(TOKEN_TABLE[state % len(TOKEN_TABLE)])
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: golden_verify.py <c_executable_path> [count]")
        return 2

    executable = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    if count <= 0:
        print("count must be a positive integer")
        return 2

    result = subprocess.run(
        [executable, str(count)], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stderr.strip() or "C program failed")
        return result.returncode

    c_tokens = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            c_tokens.append(int(stripped))
        except ValueError:
            print(f"golden verify failed: non-integer output line from C program: {stripped}")
            return 1
    expected = generate_expected(count)
    if c_tokens != expected:
        print("Golden verify failed")
        print(f"Expected: {expected}")
        print(f"Actual  : {c_tokens}")
        return 1

    print("Golden verify passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
