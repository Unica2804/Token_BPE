#!/usr/bin/env python3
import argparse
import gc
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.extend([
    str(Path(__file__).parent / "build"),
    str(Path(__file__).parent / "build" / "Release"),
    str(Path(__file__).parent),
])

try:
    import fast_bpe
except ImportError:
    print("[-] Error: 'fast_bpe' library not found. Build with cmake first.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Memory-Safe C++20 Tokenizer Benchmark")
    parser.add_argument("--file", type=Path, default=Path("shakespeare.txt"))
    parser.add_argument("--model", type=Path, default=Path("gpt2.bpe"))
    parser.add_argument("--chunk-kb", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.file.exists():
        print(f"[-] Error: File not found: {args.file}")
        sys.exit(1)

    with open(args.file, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    total_mb = len(content.encode("utf-8")) / (1024 * 1024)
    chunk_size = args.chunk_kb * 1024
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]

    print(f"[*] Loaded corpus: {total_mb:.2f} MB ({len(chunks)} chunks of ~{args.chunk_kb} KB)")

    enc = fast_bpe.BpeEncoder()
    enc.load(str(args.model))

    # Parity check on 1 sentence to verify cursor doesn't loop
    sample = "To be, or not to be, that is the question:"
    t = enc.encode(sample)
    print(f"[*] Parity sanity test: '{sample}' -> {len(t)} tokens: {t}")

    benchmarks = [
        ("C++20 Single-Thread", lambda data: [enc.encode(c) for c in data]),
        (f"C++20 Parallel ({args.threads} Threads)", lambda data: enc.encode_batch(data)),
    ]

    for name, fn in benchmarks:
        print(f"\n>>> Running: {name}")

        # Warm-up with immediate GC
        for _ in range(args.warmup):
            _ = fn(chunks)
            gc.collect()

        durations = []
        token_count = 0

        for r in range(args.runs):
            t0 = time.perf_counter()
            res = fn(chunks)
            dur = time.perf_counter() - t0
            durations.append(dur)

            if r == 0:
                token_count = sum(len(x) for x in res)

            # Do not hold old results in RAM across runs
            del res
            gc.collect()

        med_time = statistics.median(durations)
        med_speed = total_mb / med_time
        mtok_s = (token_count / med_time) / 1_000_000

        print(f"    Median Speed : {med_speed:7.2f} MB/s")
        print(f"    Throughput   : {mtok_s:7.2f} Mtok/s (Total Tokens: {token_count})")
        print(f"    Times        : {[f'{d:.4f}s' for d in durations]}")


if __name__ == "__main__":
    main()