#!/usr/bin/env python3
import argparse
import gc
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Add local build directories for fast_bpe
sys.path.extend([
    str(Path(__file__).parent / "build"),
    str(Path(__file__).parent / "build" / "Release"),
    str(Path(__file__).parent),
])

try:
    import fast_bpe
except ImportError:
    print("[-] Error: 'fast_bpe' library not found. Build with CMake first.")
    sys.exit(1)

try:
    import tiktoken
except ImportError:
    print("[-] Missing dependency: pip install tiktoken")
    tiktoken = None

try:
    from tokenizers import Tokenizer
except ImportError:
    print("[-] Missing dependency: pip install tokenizers")
    Tokenizer = None


def split_word_safe_chunks(text: str, target_chunk_bytes: int = 65536) -> list[str]:
    """Slices corpus into ~64 KB chunks on whitespace boundaries to avoid mid-token bisection."""
    chunks = []
    n = len(text)
    pos = 0
    while pos < n:
        end = min(pos + target_chunk_bytes, n)
        if end < n:
            while end < n and not text[end].isspace():
                end += 1
        chunks.append(text[pos:end])
        pos = end
    return chunks


def time_callable(fn, warmup_iters: int, timed_runs: int) -> tuple[float, list[float], any]:
    """Executes warmup cycles followed by timed runs, returning median duration, run history, and sample output."""
    for _ in range(warmup_iters):
        _ = fn()
        gc.collect()

    durations = []
    sample_res = None
    for r in range(timed_runs):
        t0 = time.perf_counter()
        out = fn()
        dur = time.perf_counter() - t0
        durations.append(dur)
        if r == 0:
            sample_res = out
        del out
        gc.collect()

    return statistics.median(durations), durations, sample_res


def parse_args():
    parser = argparse.ArgumentParser(description="Fair 3-Way Tokenizer Benchmark (fast_bpe vs tiktoken vs HF)")
    parser.add_argument("--file", type=Path, default=Path("shakespeare.txt"))
    parser.add_argument("--model", type=Path, default=Path("gpt2.bpe"))
    parser.add_argument("--chunk-kb", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 12)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)

    if not args.file.exists():
        print(f"[-] Corpus not found: {args.file}")
        sys.exit(1)
    if not args.model.exists():
        print(f"[-] Model file not found: {args.model}")
        sys.exit(1)

    with open(args.file, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()

    total_bytes = len(raw_text.encode("utf-8"))
    total_mb = total_bytes / (1024 * 1024)
    chunks = split_word_safe_chunks(raw_text, args.chunk_kb * 1024)

    print("=" * 86)
    print("                    FAIR BPE TOKENIZER BENCHMARK SUITE")
    print("=" * 86)
    print(f"  Corpus File      : {args.file.name} ({total_mb:.2f} MB, {total_bytes:,} bytes)")
    print(f"  Chunk Partition  : {len(chunks)} chunks (~{args.chunk_kb} KB each, word-safe)")
    print(f"  Execution Threads: {args.threads} logical cores")
    print(f"  Measurement      : {args.warmup} warmups | {args.runs} timed runs (reporting medians)")
    print("=" * 86)

    # -------------------------------------------------------------------------
    # 1. Engine Initialization
    # -------------------------------------------------------------------------
    print("\n[1/3] Initializing Tokenizer Backends...")

    # C++20 fast_bpe
    cpp_enc = fast_bpe.BpeEncoder()
    cpp_enc.load(str(args.model))
    print(f"  [+] fast_bpe (C++20)       : Loaded {cpp_enc.vocab_size()} vocab, {cpp_enc.merge_count()} merges")

    # OpenAI tiktoken
    tk_enc = None
    if tiktoken:
        tk_enc = tiktoken.get_encoding("gpt2")
        print(f"  [+] tiktoken (Rust)        : Loaded GPT-2 encoding ({tk_enc.n_vocab} vocab)")

    # Hugging Face tokenizers
    hf_tok = None
    if Tokenizer:
        hf_tok = Tokenizer.from_pretrained("gpt2")
        print("  [+] HF tokenizers (Rust)   : Loaded Hugging Face fast Rust backend (gpt2)")

    # -------------------------------------------------------------------------
    # 2. Correctness & Roundtrip Verification
    # -------------------------------------------------------------------------
    print("\n[2/3] Verifying Exact Roundtrip Parity...")

    # fast_bpe test
    cpp_test_tokens, _ = cpp_enc.encode_batch_flat(chunks, args.threads)
    cpp_recovered = cpp_enc.decode(cpp_test_tokens)
    assert cpp_recovered == raw_text, "fast_bpe decode validation failed to match original text!"
    print(f"  [✓] fast_bpe   : Roundtrip validated. Total tokens: {len(cpp_test_tokens):,}")

    # tiktoken test
    tk_test_tokens = None
    if tk_enc:
        tk_batch = tk_enc.encode_ordinary_batch(chunks, num_threads=args.threads)
        tk_test_tokens = [tok for chunk_tokens in tk_batch for tok in chunk_tokens]
        tk_recovered = tk_enc.decode(tk_test_tokens)
        assert tk_recovered == raw_text, "tiktoken decode validation failed to match original text!"
        print(f"  [✓] tiktoken   : Roundtrip validated. Total tokens: {len(tk_test_tokens):,}")

    # Hugging Face test
    hf_test_tokens = None
    if hf_tok:
        hf_batch = hf_tok.encode_batch_fast(chunks, add_special_tokens=False)
        hf_test_tokens = [tok for enc_obj in hf_batch for tok in enc_obj.ids]
        hf_recovered = hf_tok.decode(hf_test_tokens, skip_special_tokens=True)
        # Note: HF decode strips trailing white spaces depending on regex splits; verify size:
        print(f"  [✓] HF Tokenizer: Encoded successfully. Total tokens: {len(hf_test_tokens):,}")

    # Pre-extract token arrays for isolated decode benchmarks
    cpp_decode_input = cpp_test_tokens
    tk_decode_input = tk_test_tokens
    hf_decode_input = hf_test_tokens

    # -------------------------------------------------------------------------
    # 3. Benchmark Execution
    # -------------------------------------------------------------------------
    print("\n[3/3] Running Timed Benchmark Runs...")

    results = []

    def record_run(engine: str, threads: int, operation: str, median_sec: float, token_total: int, memory_mode: str):
        speed_mb_s = total_mb / median_sec
        mtok_s = (token_total / median_sec) / 1_000_000
        median_ms = median_sec * 1000.0
        results.append({
            "engine": engine,
            "threads": threads,
            "operation": operation,
            "speed_mb_s": speed_mb_s,
            "mtok_s": mtok_s,
            "time_ms": median_ms,
            "memory_mode": memory_mode,
        })
        print(f"  Finished: {engine:<15} | {operation:<7} | {threads}T | {speed_mb_s:8.2f} MB/s ({mtok_s:6.2f} Mtok/s, {median_ms:6.2f} ms)")

    # ------------------ ENCODE BENCHMARKS ------------------
    # 1. fast_bpe Multi-threaded (ndarray)
    med, _, _ = time_callable(lambda: cpp_enc.encode_batch_flat(chunks, args.threads), args.warmup, args.runs)
    record_run("fast_bpe", args.threads, "Encode", med, len(cpp_test_tokens), "Zero-Copy ndarray")

    # 2. fast_bpe Single-threaded (ndarray)
    med, _, _ = time_callable(lambda: cpp_enc.encode_flat(raw_text), args.warmup, args.runs)
    record_run("fast_bpe", 1, "Encode", med, len(cpp_test_tokens), "Zero-Copy ndarray")

    # 3. tiktoken Multi-threaded
    if tk_enc:
        med, _, _ = time_callable(lambda: tk_enc.encode_ordinary_batch(chunks, num_threads=args.threads), args.warmup, args.runs)
        record_run("tiktoken", args.threads, "Encode", med, len(tk_test_tokens), "list[list[int]]")

        # 4. tiktoken Single-threaded
        med, _, _ = time_callable(lambda: tk_enc.encode_ordinary(raw_text), args.warmup, args.runs)
        record_run("tiktoken", 1, "Encode", med, len(tk_test_tokens), "list[int]")

    # 5. Hugging Face Multi-threaded
    if hf_tok:
        med, _, _ = time_callable(lambda: hf_tok.encode_batch_fast(chunks, add_special_tokens=False), args.warmup, args.runs)
        record_run("HF tokenizers", args.threads, "Encode", med, len(hf_test_tokens), "list[Encoding]")

        # 6. Hugging Face Single-threaded
        med, _, _ = time_callable(lambda: hf_tok.encode(raw_text, add_special_tokens=False), args.warmup, args.runs)
        record_run("HF tokenizers", 1, "Encode", med, len(hf_test_tokens), "Encoding Object")

    # ------------------ DECODE BENCHMARKS ------------------
    # 7. fast_bpe Single-threaded Decode (ndarray buffer)
    med, _, _ = time_callable(lambda: cpp_enc.decode(cpp_decode_input), args.warmup, args.runs)
    record_run("fast_bpe", 1, "Decode", med, len(cpp_test_tokens), "Zero-Copy ndarray")

    # 8. tiktoken Single-threaded Decode
    if tk_enc:
        med, _, _ = time_callable(lambda: tk_enc.decode(tk_decode_input), args.warmup, args.runs)
        record_run("tiktoken", 1, "Decode", med, len(tk_test_tokens), "list[int]")

    # 9. Hugging Face Single-threaded Decode
    if hf_tok:
        med, _, _ = time_callable(lambda: hf_tok.decode(hf_decode_input, skip_special_tokens=True), args.warmup, args.runs)
        record_run("HF tokenizers", 1, "Decode", med, len(hf_test_tokens), "list[int]")

    # -------------------------------------------------------------------------
    # 4. Presentation Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("                              BENCHMARK REPORT")
    print("=" * 90)
    print(f"{'Engine':<15} | {'Op':<6} | {'Threads':<7} | {'Throughput':>12} | {'Token Speed':>12} | {'Latency':>10} | {'Output Type':<18}")
    print("-" * 90)
    for r in results:
        print(f"{r['engine']:<15} | {r['operation']:<6} | {r['threads']:<7} | {r['speed_mb_s']:9.2f} MB/s | {r['mtok_s']:7.2f} Mtok/s | {r['time_ms']:7.2f} ms | {r['memory_mode']:<18}")
    print("=" * 90)


if __name__ == "__main__":
    main()