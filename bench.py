import time
import sys
import os

# Point Python to your built .so / .pyd folder
sys.path.append("./build")

import fast_bpe
import tiktoken
from tokenizers import Tokenizer

def run_benchmark():
    print("Preparing corpora and tokenizers...")

    # 1. Initialize Engines
    # Load our C++ engine
    custom_enc = fast_bpe.BpeEncoder()
    # Assuming 'gpt2.bpe' has been saved with GPT-2's vocabulary & merges
    custom_enc.load("gpt2.bpe")

    # Load tiktoken (GPT-2 encoding)
    tik_enc = tiktoken.get_encoding("gpt2")

    # Load Hugging Face fast tokenizer (Rust backend)
    hf_enc = Tokenizer.from_pretrained("gpt2")

    # 2. Build Test Dataset (~10 MB of text across 5,000 chunks)
    sample_paragraph = (
        "Transformer models rely on self-attention mechanisms to capture long-range "
        "dependencies across sequences. Byte Pair Encoding iteratively merges the most "
        "frequently occurring adjacent character pairs into single atomic tokens. "
        "High performance systems leverage zero-copy views and SIMD vectorization.\n"
    )
    num_chunks = 5000
    chunks = [sample_paragraph * 4 for _ in range(num_chunks)]
    
    total_bytes = sum(len(c.encode("utf-8")) for c in chunks)
    total_mb = total_bytes / (1024 * 1024)
    print(f"Total benchmark corpus size: {total_mb:.2f} MB ({num_chunks} chunks)\n")

    # Warm-up phase
    _ = custom_enc.encode(chunks[0])
    _ = tik_enc.encode(chunks[0])
    _ = hf_enc.encode(chunks[0])

    # -------------------------------------------------------------
    # Test 1: Single-Threaded Throughput (Chunk-by-chunk iteration)
    # -------------------------------------------------------------
    print("--- [1] Single-Threaded Evaluation ---")

    # Custom C++ Tokenizer
    t0 = time.perf_counter()
    custom_tokens = [custom_enc.encode(c) for c in chunks]
    t_custom_st = time.perf_counter() - t0
    custom_mb_s = total_mb / t_custom_st
    print(f"Custom C++20 Tokenizer : {t_custom_st:.3f} s | {custom_mb_s:.2f} MB/s")

    # OpenAI tiktoken
    t0 = time.perf_counter()
    tik_tokens = [tik_enc.encode(c) for c in chunks]
    t_tik = time.perf_counter() - t0
    tik_mb_s = total_mb / t_tik
    print(f"OpenAI tiktoken        : {t_tik:.3f} s | {tik_mb_s:.2f} MB/s")

    # Hugging Face tokenizers
    t0 = time.perf_counter()
    hf_tokens = [hf_enc.encode(c).ids for c in chunks]
    t_hf_st = time.perf_counter() - t0
    hf_mb_s = total_mb / t_hf_st
    print(f"Hugging Face Tokenizers: {t_hf_st:.3f} s | {hf_mb_s:.2f} MB/s")

    # -------------------------------------------------------------
    # Test 2: Multi-Threaded Batch Throughput
    # -------------------------------------------------------------
    # Use the machine's hardware thread count across all three
    num_threads = os.cpu_count() or 8

    print(
        f"\n--- [2] Multi-Threaded Batch Evaluation ({num_threads} threads) ---"
    )

    # 1. Custom C++20 Tokenizer (std::jthread / OpenMP)
    t0 = time.perf_counter()
    _ = custom_enc.encode_batch(chunks)
    t_custom_mt = time.perf_counter() - t0
    print(
        f"Custom C++20 (Batch)    : {t_custom_mt:.3f} s | {total_mb / t_custom_mt:.2f} MB/s"
    )

    # 2. OpenAI tiktoken (encode_ordinary_batch in Rust)
    t0 = time.perf_counter()
    _ = tik_enc.encode_ordinary_batch(chunks, num_threads=num_threads)
    t_tik_mt = time.perf_counter() - t0
    print(
        f"OpenAI tiktoken (Batch) : {t_tik_mt:.3f} s | {total_mb / t_tik_mt:.2f} MB/s"
    )

    # 3. Hugging Face Tokenizers (Rayon thread-pool in Rust)
    t0 = time.perf_counter()
    _ = hf_enc.encode_batch(chunks)
    t_hf_mt = time.perf_counter() - t0
    print(
        f"Hugging Face (Batch)    : {t_hf_mt:.3f} s | {total_mb / t_hf_mt:.2f} MB/s"
    )

if __name__ == "__main__":
    run_benchmark()