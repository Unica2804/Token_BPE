#!/usr/bin/env python3
import argparse
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

# Add C++ build artifact path (handles both in-tree and build/ directory)
sys.path.extend([
    str(Path(__file__).parent / "build"),
    str(Path(__file__).parent / "build" / "Release"),
    str(Path(__file__).parent)
])

try:
    import fast_bpe
except ImportError:
    print("[-] Error: 'fast_bpe' module not found. Build your C++ extension first:")
    print("    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release")
    sys.exit(1)

try:
    import tiktoken
except ImportError:
    print("[-] Error: 'tiktoken' is not installed. Run: pip install tiktoken")
    sys.exit(1)

try:
    from tokenizers import Tokenizer
except ImportError:
    print("[-] Error: 'tokenizers' is not installed. Run: pip install tokenizers")
    sys.exit(1)


DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Rigorous Tokenizer Benchmark Harness")
    parser.add_argument("--file", type=Path, default=None, help="Path to text corpus (defaults to shakespeare.txt)")
    parser.add_argument("--model", type=Path, default=Path("gpt2.bpe"), help="Path to custom gpt2.bpe model")
    parser.add_argument("--chunk-kb", type=int, default=64, help="Chunk size in KB (default: 64 KB to saturate worker threads)")
    parser.add_argument("--warmup", type=int, default=3, help="Number of untimed warmup iterations")
    parser.add_argument("--runs", type=int, default=7, help="Number of timed benchmark runs")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 8, help="Thread count for multi-threaded tests")
    return parser.parse_args()


def ensure_corpus(file_path: Path | None) -> Path:
    if file_path and file_path.exists():
        return file_path
    
    target = Path("shakespeare.txt")
    if not target.exists():
        print(f"[*] Downloading benchmark corpus from: {DEFAULT_CORPUS_URL}")
        urllib.request.urlretrieve(DEFAULT_CORPUS_URL, target)
        print(f"[+] Saved corpus to: {target.resolve()}")
    return target


def load_chunks(corpus_path: Path, chunk_kb: int) -> tuple[list[str], float]:
    print(f"[*] Reading corpus: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    total_bytes = len(content.encode("utf-8"))
    total_mb = total_bytes / (1024 * 1024)

    chunk_size = chunk_kb * 1024
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]

    print(f"    Total Size: {total_mb:.2f} MB split into {len(chunks)} chunks of ~{chunk_kb} KB each.")
    return chunks, total_mb


def verify_token_parity(custom_enc, tik_enc, test_sentence: str):
    print("\n[*] Checking Token Parity on sample sentence...")
    print(f"    Sample: \"{test_sentence}\"")
    
    custom_tokens = custom_enc.encode(test_sentence)
    tik_tokens = tik_enc.encode_ordinary(test_sentence)

    print(f"    Custom Engine Tokens ({len(custom_tokens)}): {custom_tokens[:12]}")
    print(f"    Tiktoken Tokens      ({len(tik_tokens)}): {tik_tokens[:12]}")

    if custom_tokens == tik_tokens:
        print("    [✓] Parity Verified: Output token IDs match exactly.\n")
    else:
        print("    [!] Parity Warning: Token counts or IDs differ.")
        print("        Explanation: If custom output length is higher, custom BPE lacks regex space-prefixing (Ġ),")
        print("        which affects compression ratio and tokens/sec metrics.\n")


def measure_engine(name: str, encode_fn, chunks: list[str], total_mb: float, warmup: int, runs: int) -> dict:
    print(f">>> Benchmarking: {name} (Warmup: {warmup}, Timed Runs: {runs})")

    # 1. Warm-up passes (pin clock speed, warm OS page caches)
    for _ in range(warmup):
        _ = encode_fn(chunks)

    # 2. Timed runs
    durations = []
    total_tokens = 0

    for r in range(runs):
        t0 = time.perf_counter()
        results = encode_fn(chunks)
        dur = time.perf_counter() - t0
        durations.append(dur)

        if r == 0:
            # Determine total token count
            if hasattr(results[0], "ids"):  # Hugging Face Encoding object
                total_tokens = sum(len(x.ids) for x in results)
            elif isinstance(results[0], (list, tuple)):
                total_tokens = sum(len(x) for x in results)
            else:
                total_tokens = len(results)

    # 3. Aggregate statistics
    throughputs = [total_mb / d for d in durations]
    median_speed = statistics.median(throughputs)
    mean_speed = statistics.mean(throughputs)
    stdev_speed = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0

    median_duration = statistics.median(durations)
    mtok_s = (total_tokens / median_duration) / 1_000_000

    print(f"    Median Throughput : {median_speed:7.2f} MB/s")
    print(f"    Mean ± StDev      : {mean_speed:7.2f} ± {stdev_speed:.2f} MB/s")
    print(f"    Token Throughput  : {mtok_s:7.2f} Mtok/s (Total Tokens: {total_tokens})")
    print(f"    Run Times         : {[f'{d:.4f}s' for d in durations]}\n")

    return {
        "name": name,
        "median_mbs": median_speed,
        "mean_mbs": mean_speed,
        "stdev": stdev_speed,
        "mtok_s": mtok_s,
        "tokens": total_tokens
    }


def main():
    args = parse_args()

    corpus_path = ensure_corpus(args.file)
    chunks, total_mb = load_chunks(corpus_path, args.chunk_kb)

    # Initialize Tokenizers
    print("[*] Initializing Tokenizers...")
    custom_enc = fast_bpe.BpeEncoder()
    if not args.model.exists():
        print(f"[-] Error: Model file '{args.model}' not found.")
        print("    Convert Hugging Face GPT-2 vocab first: python3 convert_hf_to_bpe.py")
        sys.exit(1)
    custom_enc.load(str(args.model))

    tik_enc = tiktoken.get_encoding("gpt2")
    hf_enc = Tokenizer.from_pretrained("gpt2")

    # Parity check
    verify_token_parity(custom_enc, tik_enc, "To be, or not to be, that is the question:")

    records = []

    # -------------------------------------------------------------
    # 1. Single-Threaded Benchmarks
    # -------------------------------------------------------------
    print("========================================================")
    print("             [1] SINGLE-THREADED BENCHMARKS             ")
    print("========================================================")

    # Custom C++ (Uncached Scratchpad)
    if hasattr(custom_enc, "encode_uncached"):
        records.append(measure_engine(
            "Custom C++20 (Uncached)",
            lambda data: [custom_enc.encode_uncached(c) for c in data],
            chunks, total_mb, args.warmup, args.runs
        ))

    # Custom C++ (Memoized Cache)
    records.append(measure_engine(
        "Custom C++20 (Memoized Cache)",
        lambda data: [custom_enc.encode(c) for c in data],
        chunks, total_mb, args.warmup, args.runs
    ))

    # OpenAI tiktoken (Single-thread)
    records.append(measure_engine(
        "OpenAI tiktoken (Single-Thread)",
        lambda data: [tik_enc.encode_ordinary(c) for c in data],
        chunks, total_mb, args.warmup, args.runs
    ))

    # Hugging Face tokenizers (Single-thread)
    records.append(measure_engine(
        "Hugging Face (Single-Thread)",
        lambda data: [hf_enc.encode(c).ids for c in data],
        chunks, total_mb, args.warmup, args.runs
    ))

    # -------------------------------------------------------------
    # 2. Multi-Threaded Batch Benchmarks
    # -------------------------------------------------------------
    print("========================================================")
    print(f"         [2] MULTI-THREADED BATCH ({args.threads} Threads)        ")
    print("========================================================")

    # Custom C++20 Parallel (Batch)
    records.append(measure_engine(
        f"Custom C++20 (Parallel {args.threads}T)",
        lambda data: custom_enc.encode_batch(data),
        chunks, total_mb, args.warmup, args.runs
    ))

    # OpenAI tiktoken Parallel (Batch)
    records.append(measure_engine(
        f"OpenAI tiktoken ({args.threads}T Batch)",
        lambda data: tik_enc.encode_ordinary_batch(data, num_threads=args.threads),
        chunks, total_mb, args.warmup, args.runs
    ))

    # Hugging Face Parallel (Rayon Batch)
    records.append(measure_engine(
        "Hugging Face (Rayon Batch)",
        lambda data: hf_enc.encode_batch(data),
        chunks, total_mb, args.warmup, args.runs
    ))

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    print("=========================================================================================")
    print("                                   FINAL COMPARISON                                      ")
    print("=========================================================================================")
    print(f"{'Engine':<35} | {'Median MB/s':>12} | {'Mean ± StDev':>18} | {'Mtok/s':>10}")
    print("-" * 89)
    for r in records:
        print(f"{r['name']:<35} | {r['median_mbs']:>12.2f} | {r['mean_mbs']:>8.2f} ± {r['stdev']:<6.2f} | {r['mtok_s']:>10.2f}")
    print("=========================================================================================")


if __name__ == "__main__":
    main()