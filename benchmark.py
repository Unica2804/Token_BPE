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

import fast_bpe

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None


def split_chunks_on_newlines(text: str, target_bytes: int = 65536) -> list[str]:
    """
    Splits corpus strictly AFTER newline ('\\n') characters.
    In GPT-2 BPE, tokens never merge across a newline boundary when followed by another line,
    avoiding mid-word and mid-indentation splits.
    """
    chunks = []
    n = len(text)
    pos = 0
    while pos < n:
        end = min(pos + target_bytes, n)
        if end < n:
            nl = text.find("\n", end)
            end = (nl + 1) if nl != -1 else n
        chunks.append(text[pos:end])
        pos = end
    return chunks


def run_timed(fn, warmup: int, runs: int):
    for _ in range(warmup):
        _ = fn()
        gc.collect()

    durations = []
    for _ in range(runs):
        t0 = time.perf_counter()
        res = fn()
        dur = time.perf_counter() - t0
        durations.append(dur)
        del res
        gc.collect()

    return statistics.median(durations), min(durations)


def main():
    parser = argparse.ArgumentParser(description="Audited Fair 3-Way BPE Benchmark")
    parser.add_argument("--file", type=Path, default=Path("shakespeare.txt"))
    parser.add_argument("--model", type=Path, default=Path("gpt2.bpe"))
    parser.add_argument("--chunk-kb", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    args = parser.parse_args()

    # Pin Rust Rayon thread pool to match hardware concurrency used by PersistentThreadPool
    hw_threads = os.cpu_count() or args.threads
    os.environ["RAYON_NUM_THREADS"] = str(hw_threads)

    raw_text = args.file.read_text(encoding="utf-8", errors="replace")
    total_bytes = len(raw_text.encode("utf-8"))
    total_mb = total_bytes / (1024 * 1024)
    chunks = split_chunks_on_newlines(raw_text, args.chunk_kb * 1024)

    print("=" * 94)
    print("                     AUDITED 3-WAY BPE BENCHMARK (GPT-2 VOCABULARY)")
    print("=" * 94)
    print(f"  Corpus         : {args.file.name} ({total_mb:.2f} MB | {total_bytes:,} bytes)")
    print(f"  Chunks         : {len(chunks)} newline-aligned chunks (~{args.chunk_kb} KB each)")
    print(f"  Thread Pool    : {hw_threads} hardware threads (C++ PersistentThreadPool & Rust Rayon)")
    print("=" * 94)

    cpp_enc = fast_bpe.BpeEncoder()
    cpp_enc.load(str(args.model))

    tk_enc = tiktoken.get_encoding("gpt2") if tiktoken else None
    hf_tok = Tokenizer.from_pretrained("gpt2") if Tokenizer else None

    # Pre-compute outputs once outside timers to verify roundtrip & prepare decode inputs
    cpp_flat_np, _ = cpp_enc.encode_batch_flat(chunks, hw_threads)
    cpp_flat_list = cpp_flat_np.tolist()
    assert cpp_enc.decode(cpp_flat_np) == raw_text, "fast_bpe ndarray roundtrip mismatch!"
    assert cpp_enc.decode(cpp_flat_list) == raw_text, "fast_bpe list[int] roundtrip mismatch!"

    if tk_enc:
        tk_nested = tk_enc.encode_ordinary_batch(chunks, num_threads=hw_threads)
        tk_flat_list = [t for sub in tk_nested for t in sub]
        assert tk_enc.decode(tk_flat_list) == raw_text, "tiktoken roundtrip mismatch!"

    if hf_tok:
        hf_enc_objs = hf_tok.encode_batch_fast(chunks, add_special_tokens=False)
        hf_flat_list = [t for obj in hf_enc_objs for t in obj.ids]

    rows = []

    def bench(label: str, op: str, threads: int, tok_count: int, mode: str, fn):
        med_s, min_s = run_timed(fn, args.warmup, args.runs)
        mb_s = total_mb / med_s
        mtok_s = (tok_count / med_s) / 1e6
        rows.append((label, op, f"{threads}T", mb_s, mtok_s, med_s * 1000, tok_count, mode))

    # --- 1. MULTI-THREADED BATCH ENCODE (64KB Chunks) ---
    bench("fast_bpe (C++20)", "Encode", hw_threads, len(cpp_flat_np), "Zero-Copy ndarray",
          lambda: cpp_enc.encode_batch_flat(chunks, hw_threads))
    bench("fast_bpe (C++20)", "Encode", hw_threads, len(cpp_flat_np), "Python list[list[int]]",
          lambda: cpp_enc.encode_batch(chunks))
    if tk_enc:
        bench("tiktoken (Rust)", "Encode", hw_threads, len(tk_flat_list), "Python list[list[int]]",
              lambda: tk_enc.encode_ordinary_batch(chunks, num_threads=hw_threads))
    if hf_tok:
        bench("HF tokenizers", "Encode", hw_threads, len(hf_flat_list), "Rust Encoding Batch",
              lambda: hf_tok.encode_batch_fast(chunks, add_special_tokens=False))

    # --- 2. SINGLE-THREADED ENCODE (Same 64KB Chunks, 1 Thread) ---
    bench("fast_bpe (C++20)", "Encode", 1, len(cpp_flat_np), "Python list[list[int]]",
          lambda: [cpp_enc.encode(c) for c in chunks])
    if tk_enc:
        bench("tiktoken (Rust)", "Encode", 1, len(tk_flat_list), "Python list[list[int]]",
              lambda: [tk_enc.encode_ordinary(c) for c in chunks])
    if hf_tok:
        bench("HF tokenizers", "Encode", 1, len(hf_flat_list), "Rust Encoding List",
              lambda: [hf_tok.encode(c, add_special_tokens=False) for c in chunks])

    # --- 3. SINGLE-THREADED DECODE (Both ndarray and Apples-to-Apples list[int]) ---
    bench("fast_bpe (C++20)", "Decode", 1, len(cpp_flat_np), "Zero-Copy ndarray",
          lambda: cpp_enc.decode(cpp_flat_np))
    bench("fast_bpe (C++20)", "Decode", 1, len(cpp_flat_list), "Python list[int]",
          lambda: cpp_enc.decode(cpp_flat_list))
    if tk_enc:
        bench("tiktoken (Rust)", "Decode", 1, len(tk_flat_list), "Python list[int]",
              lambda: tk_enc.decode(tk_flat_list))
    if hf_tok:
        bench("HF tokenizers", "Decode", 1, len(hf_flat_list), "Python list[int]",
              lambda: hf_tok.decode(hf_flat_list, skip_special_tokens=True))

    print("\n" + "=" * 106)
    print(f"{'Engine':<18} | {'Op':<6} | {'Thr':<3} | {'Median Speed':>13} | {'Throughput':>12} | {'Latency':>9} | {'Tokens':>9} | {'FFI Boundary Type'}")
    print("-" * 106)
    for r in rows:
        print(f"{r[0]:<18} | {r[1]:<6} | {r[2]:<3} | {r[3]:9.2f} MB/s | {r[4]:6.2f} Mtok/s | {r[5]:6.2f} ms | {r[6]:9,d} | {r[7]}")
    print("=" * 106)


if __name__ == "__main__":
    main()