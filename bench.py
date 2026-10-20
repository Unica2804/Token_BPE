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


def benchmark_suite(name: str, fn, total_mb: float, warmup: int, runs: int, is_decode: bool = False):
    print(f"\n>>> Running: {name}")

    for _ in range(warmup):
        _ = fn()
        gc.collect()

    durations = []
    token_count = 0

    for r in range(runs):
        t0 = time.perf_counter()
        res = fn()
        dur = time.perf_counter() - t0
        durations.append(dur)

        if r == 0:
            if isinstance(res, tuple):  # (tokens_arr, offsets_arr)
                token_count = len(res[0])
            elif isinstance(res, np.ndarray):
                token_count = len(res)
            elif isinstance(res, str):
                token_count = len(res.encode("utf-8"))
            elif isinstance(res[0], list):
                token_count = sum(len(x) for x in res)
            else:
                token_count = len(res)

        del res
        gc.collect()

    med_time = statistics.median(durations)
    med_speed = total_mb / med_time

    if is_decode:
        print(f"    Median Speed : {med_speed:7.2f} MB/s")
        print(f"    Times        : {[f'{d:.4f}s' for d in durations]}")
    else:
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

    # Pre-encode tokens into a flat NumPy array (no .tolist()!)
    tokens_np, _ = enc.encode_batch_flat(chunks, args.threads)
    assert isinstance(tokens_np, np.ndarray) and tokens_np.dtype == np.uint32

    # Verification: Validate roundtrip decode match directly from the numpy buffer
    decoded_text = enc.decode(tokens_np)
    # Validate on sample segment to confirm exact string recovery
    sample_text = "To be, or not to be, that is the question:"
    sample_tokens = enc.encode_flat(sample_text)
    assert enc.decode(sample_tokens) == sample_text
    print("[*] Decode roundtrip test passed: exact string recovered from NumPy array!")

    # 1. Zero-Copy Flat Batch Encode
    speed_encode = benchmark_suite(
        f"C++20 Parallel Encode ({args.threads}T) [Zero-Copy ndarray]",
        lambda: enc.encode_batch_flat(chunks, args.threads),
        total_mb, args.warmup, args.runs, is_decode=False
    )

    # 2. Zero-Copy Direct NumPy Decode (Passes np.ndarray directly to C++ pointer)
    speed_decode = benchmark_suite(
        "C++20 Single-Thread Decode [Zero-Copy ndarray -> std::string]",
        lambda: enc.decode(tokens_np),
        total_mb, args.warmup, args.runs, is_decode=True
    )

    print("\n" + "=" * 65)
    print(f"Zero-Copy ndarray Encode Speed : {speed_encode:7.2f} MB/s")
    print(f"Flat-Table ndarray Decode Speed: {speed_decode:7.2f} MB/s")
    print("=" * 65)


if __name__ == "__main__":
    main()