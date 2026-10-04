#!/usr/bin/env python3
"""Compare the fast Walsh reference with the frozen direct implementation.

Runs actual source modules without importing the wider quantum package. No
PyQPanda installation is needed. This measures classical reference evaluation,
not simulator performance. The original commit must be locally available unless
--baseline-file supplies its oracle_learning.py bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = Path("pyqpanda-algorithm/pyqpanda_alg/QOracleLearning/oracle_learning.py")
BASELINE_COMMIT = "b04476c0e4486d8b6e3bc503965d8fd1dac9192d"
BASELINE_BLOB = "b12c975f3b0d526c4e587052d64eb0a1e376b02d"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(function: Any, table: tuple[int, ...]) -> tuple[tuple[float, ...], float]:
    started = time.perf_counter_ns()
    result = function(table)
    return result, (time.perf_counter_ns() - started) / 1_000_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path)
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--output", type=Path, default=Path("walsh-benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 100:
        parser.error("repeats must be in 1..100")
    if not args.widths or any(n < 1 or n > 14 for n in args.widths):
        parser.error("widths must be in 1..14; the baseline has quadratic cost")
    try:
        baseline_bytes = (
            args.baseline_file.read_bytes() if args.baseline_file else
            subprocess.check_output(
                ["git", "show", f"{BASELINE_COMMIT}:{MODULE_PATH.as_posix()}"], cwd=ROOT,
            )
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        parser.error(f"Read the frozen source locally or provide --baseline-file: {exc}")
    if git_blob(baseline_bytes) != BASELINE_BLOB:
        parser.error("baseline file does not match the frozen contribution's Git blob")

    candidate_path = ROOT / MODULE_PATH
    candidate_bytes = candidate_path.read_bytes()
    with tempfile.TemporaryDirectory(prefix="walsh-baseline-") as temporary:
        baseline_path = Path(temporary) / "oracle_learning.py"
        baseline_path.write_bytes(baseline_bytes)
        original = load_module(baseline_path, "walsh_original")
        candidate = load_module(candidate_path, "walsh_candidate")
        exhaustive_count = 0
        for size in (2, 4, 8):
            for table in itertools.product((0, 1), repeat=size):
                if candidate.walsh_probabilities(table) != original.walsh_probabilities(table):
                    raise AssertionError(f"Output mismatch on {table}")
                exhaustive_count += 1

        cases = []
        for width in args.widths:
            size = 1 << width
            rng = random.Random(args.seed + width)
            table = tuple(rng.randrange(2) for _ in range(size))
            reference = original.walsh_probabilities(table)
            if candidate.walsh_probabilities(table) != reference:
                raise AssertionError(f"Warm-up mismatch at width {width}")
            old_samples, new_samples = [], []
            for repetition in range(args.repeats):
                # Alternate order to reduce a consistent first-run timing bias.
                functions = [original.walsh_probabilities, candidate.walsh_probabilities]
                order = (0, 1) if repetition % 2 == 0 else (1, 0)
                for which in order:
                    result, seconds = timed(functions[which], table)
                    if result != reference:
                        raise AssertionError(f"Output mismatch at width {width}, repeat {repetition}")
                    (old_samples if which == 0 else new_samples).append(seconds)
            old_median = statistics.median(old_samples)
            new_median = statistics.median(new_samples)
            row = {
                "width": width, "truth_table_entries": size,
                "input_sha256": hashlib.sha256(bytes(table)).hexdigest(),
                "baseline_seconds": old_samples, "candidate_seconds": new_samples,
                "baseline_median_seconds": old_median,
                "candidate_median_seconds": new_median,
                "median_speedup": old_median / new_median,
                "exact_output_equal": True, "max_absolute_difference": 0.0,
            }
            cases.append(row)
            print(f"{size:5d} entries: {old_median:.6f}s -> {new_median:.6f}s "
                  f"({row['median_speedup']:.2f}x), exactly equal")

    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu = next((line.partition(":")[2].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    report = {
        "measurement": "single-process classical Walsh reference; not quantum simulator speed",
        "baseline_commit": BASELINE_COMMIT, "baseline_git_blob": BASELINE_BLOB,
        "candidate_git_blob": git_blob(candidate_bytes),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version, "platform": platform.platform(), "cpu": cpu,
        "clock": "time.perf_counter_ns", "repeats": args.repeats,
        "seed": args.seed, "exhaustive_boolean_tables": exhaustive_count,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {exhaustive_count} exhaustive small tables; wrote {args.output}")


if __name__ == "__main__":
    main()
