#!/usr/bin/env python3
import argparse
import gc
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

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
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    return parser.parse_args()


def benchmark_suite(name: str, fn, chunks: list[str], total_mb: float, warmup: int, runs: int):
    print(f"\n>>> Running: {name}")

    for _ in range(warmup):
        _ = fn(chunks)
        gc.collect()

    durations = []
    token_count = 0

    for r in range(runs):
        t0 = time.perf_counter()
        res = fn(chunks)
        dur = time.perf_counter() - t0
        durations.append(dur)

        if r == 0:
            if isinstance(res, tuple):  # (tokens_arr, offsets_arr)
                token_count = len(res[0])
            elif isinstance(res, np.ndarray):
                token_count = len(res)
            elif isinstance(res[0], list):
                token_count = sum(len(x) for x in res)
            else:
                token_count = len(res)

        del res
        gc.collect()

    med_time = statistics.median(durations)
    med_speed = total_mb / med_time
    mtok_s = (token_count / med_time) / 1_000_000

    print(f"    Median Speed : {med_speed:7.2f} MB/s")
    print(f"    Throughput   : {mtok_s:7.2f} Mtok/s (Tokens: {token_count})")
    print(f"    Times        : {[f'{d:.4f}s' for d in durations]}")

    return med_speed


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

    # Verify zero-copy output type
    tokens, offsets = enc.encode_batch_flat(chunks[:2], args.threads)
    assert isinstance(tokens, np.ndarray) and tokens.dtype == np.uint32
    assert isinstance(offsets, np.ndarray) and offsets.dtype == np.uint32
    print(f"[*] Zero-copy validation passed: tokens array dtype={tokens.dtype}, shape={tokens.shape}")

    # 1. Standard Multi-threaded list[list[int]]
    speed_legacy = benchmark_suite(
        f"C++20 Parallel ({args.threads}T) [list[list[int]]]",
        lambda data: enc.encode_batch(data),
        chunks, total_mb, args.warmup, args.runs
    )

    # 2. Zero-Copy Flat Buffer Protocol
    speed_zero_copy = benchmark_suite(
        f"C++20 Parallel ({args.threads}T) [Zero-Copy ndarray]",
        lambda data: enc.encode_batch_flat(data, args.threads),
        chunks, total_mb, args.warmup, args.runs
    )

    print("\n" + "=" * 65)
    print(f"Legacy Python Marshaling Throughput : {speed_legacy:7.2f} MB/s")
    print(f"Zero-Copy ndarray Throughput         : {speed_zero_copy:7.2f} MB/s")
    print(f"Speedup from Eliminating Heap Alloc : {speed_zero_copy / speed_legacy:7.2f}x")
    print("=" * 65)


if __name__ == "__main__":
    main()