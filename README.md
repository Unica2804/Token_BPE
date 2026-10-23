# BPE Tokenizer

A fast **Byte Pair Encoding** tokenizer written in **C++20** (zero runtime
dependencies), with a **nanobind** Python module (`fast_bpe`), a built-in
trainer, a compact binary model format, SWAR-accelerated pre-tokenization, a
persistent thread pool, and zero-copy NumPy bindings.

---

## Features

- **SWAR pre-tokenizer** — a GPT-2-style splitter that classifies 8 bytes per
  instruction (letters, digits, whitespace, contractions, punctuation) instead
  of using a regex engine.
- **Dual-cursor encoding** — long chunks are split at a safe whitespace
  boundary near the midpoint and both halves are advanced alternately, exposing
  instruction-level parallelism and improving cache reuse.
- **Intrusive linked-list BPE** — each pass selects the globally lowest-rank
  pair in O(n) and merges it in place with O(1) unlink; no string allocations
  during merging.
- **Thread-local pre-token cache** — 8 192 direct-mapped 64-byte slots (512 KiB)
  with FNV-1a hashing; caches pre-tokens up to 22 bytes / 8 tokens.
- **Persistent thread pool** — reusable workers (`hardware_concurrency() - 1`
  plus the caller) instead of spawning `std::jthread`s per batch, so thread-local
  caches and scratchpads stay warm across calls.
- **Thread-local scratchpad** — reusable merge-node and output buffers; the
  streaming `*_to_buffer` APIs append into caller-owned vectors.
- **Zero-copy flat batches** — `parallel_tokenize_flat()` returns one contiguous
  token array plus an offsets array, eliminating per-chunk vectors and copies.
- **NumPy fast paths** — `encode_flat` / `encode_batch_flat` return
  `numpy.uint32` arrays (ownership transferred via a capsule); `decode` accepts
  a 1-D contiguous `uint32` ndarray directly. The GIL is released around every
  compute phase.
- **Flat decode table** — decoding is a single allocation plus O(1) table
  lookups and `memcpy` per token.
- **Trainer** — frequency-weighted BPE training from raw byte corpora.
- **Binary model format (`BPE1`)** — compact, endian-explicit, loads in
  microseconds (GPT-2: 50 257 tokens / 49 992 merges ≈ 1.5 MB).
- **HF importer** — convert Hugging Face `vocab.json` + `merges.txt` into the
  `.bpe` binary.
- **Audited 3-way benchmark** — `benchmark.py` compares `fast_bpe` against
  OpenAI `tiktoken` and Hugging Face `tokenizers` on identical newline-aligned
  chunks, verifies `fast_bpe`/`tiktoken` roundtrips, and pins Rust's Rayon pool
  to the same thread count as the C++ pool.
- **Tests** — Catch2 v3 suite (10 cases, 52 assertions) wired into CTest.

---

## Repository layout

```
BPE_Tokenizer/
├── CMakeLists.txt              # Build system (C++20, nanobind, Catch2, CTest)
├── main.cpp                    # Demo executable -> build/tokenizer_cli
├── benchmark.py                # Audited 3-way Python benchmark (fast_bpe/tiktoken/HF)
├── bench.py                    # Earlier fast_bpe-only zero-copy benchmark
├── bench/benchmark.cpp         # C++ benchmark suite -> build/tokenizer_bench
├── include/tokenizer/
│   ├── types.hpp               # TokenId, TokenPair, hash
│   ├── encoder.hpp             # BpeEncoder
│   ├── trainer.hpp             # BpeTrainer / WordItem
│   ├── parallel.hpp            # parallel_tokenize(), FlatBatch, parallel_tokenize_flat()
│   ├── pre_tokenizer.hpp       # LexerCursor + next_token() GPT-2-style splitter
│   ├── swar.hpp                # 8-byte-at-a-time ASCII classification primitives
│   ├── dual_cursor.hpp         # midpoint split + fused dual-cursor driver
│   ├── token_cache.hpp         # thread-local 8 192-slot pre-token cache
│   ├── scratchpad.hpp          # thread-local ListNode / output buffers
│   └── thread_pool.hpp         # PersistentThreadPool singleton
├── src/
│   ├── encoder.cpp             # encode / decode / save / load
│   └── trainer.cpp             # training loop
├── utils/
│   ├── bindings.cpp            # nanobind module "fast_bpe"
│   └── convert_hf_to_bpe.py    # HF tokenizer -> gpt2.bpe converter
├── tests/
│   ├── test_encoder.cpp        # [encoder] [cache] [scratchpad] [serialization]
│   └── test_trainer.cpp        # [trainer]
├── gpt2_tokenizer/             # Upstream GPT-2 vocab.json + merges.txt
├── shakespeare.txt             # Benchmark corpus (~1.06 MB)
└── gpt2.bpe                    # Pre-converted binary model (50 257 tokens)
```

### Header reference

| Header | Contents |
|---|---|
| `types.hpp` | `TokenId`, `INVALID_TOKEN_ID`, `TokenPair`, `TokenPairHash` |
| `encoder.hpp` | `BpeEncoder`: encode, buffer, decode, save/load, merge accessors |
| `trainer.hpp` | `BpeTrainer::train()`, `WordItem` |
| `parallel.hpp` | `parallel_tokenize()`, `parallel_tokenize_flat()`, `FlatBatch` |
| `pre_tokenizer.hpp` | `LexerCursor`, `next_token()` |
| `swar.hpp` | `all_ascii_letters`, `all_ascii_digits`, `all_bytes_in_range`, `load64` |
| `dual_cursor.hpp` | `find_safe_split_boundary()`, `process_dual_cursor()` |
| `token_cache.hpp` | `TokenCache` (per-thread, direct-mapped) |
| `scratchpad.hpp` | `ThreadScratchpad`, `g_scratchpad`, `ListNode` |
| `thread_pool.hpp` | `PersistentThreadPool::instance().parallel_for()` |

---

## Requirements

| Component | Version |
|---|---|
| C++ compiler | C++20 (GCC, Clang, MSVC) |
| CMake | ≥ 3.20 |
| Python | ≥ 3.10 (with development headers) |
| Python packages | `nanobind` (build), `numpy` (Python benchmarks and ndarray fast paths) |
| Benchmark packages | `tiktoken` + `tokenizers` (optional, 3-way comparison) |
| Network | First configure (Catch2 via `FetchContent`); first `benchmark.py` run (downloads the GPT-2 encoding for `tiktoken` / Hugging Face) |

```bash
python -m venv .venv
source .venv/bin/activate
pip install nanobind numpy

# Optional: enables the tiktoken / Hugging Face columns
pip install tiktoken tokenizers
```

---

## Building

```bash
# Configure (picks up .venv/bin/python automatically for nanobind)
cmake -S . -B build

# Build everything: core lib, CLI, benchmarks, tests, Python module
cmake --build build -j
```

Artifacts produced in `build/`:

| Artifact | Description |
|---|---|
| `libbpe_core.a` | Static library with the tokenizer core |
| `tokenizer_cli` | Demo executable |
| `tokenizer_bench` | C++ benchmark suite |
| `tokenizer_tests` | Catch2 test binary |
| `fast_bpe.*.so` | Python extension module |

> **Note:** GCC and Clang builds use `-O3 -march=native -flto -DNDEBUG`
> unconditionally, and `CMAKE_BUILD_TYPE` defaults to `Release`. MSVC builds use
> `/O2 /W4 /permissive-`.

---

## Usage

### C++ API

```cpp
#include "tokenizer/encoder.hpp"
#include "tokenizer/trainer.hpp"
#include "tokenizer/parallel.hpp"

using namespace tokenizer;

// --- Train from a byte-frequency corpus -------------------------------
BpeEncoder encoder;
std::vector<WordItem> corpus = {
    {{'h','u','g'},     10},
    {{'p','u','g'},      5},
    {{'h','u','g','s'}, 12},
    {{'b','u','g','s'},  4}
};
BpeTrainer::train(corpus, /*target_vocab_size=*/262, encoder);

// --- Encode / decode --------------------------------------------------
std::vector<TokenId> ids = encoder.encode_chunk("hugs");
std::string text         = encoder.decode(ids);

// --- Streaming: reuse caller-owned buffers ----------------------------
std::vector<TokenId> out;
encoder.encode_chunk_to_buffer("hugs", out);          // cache + merge + append

// --- Bypass the pre-token cache ----------------------------------------
std::vector<TokenId> raw = encoder.encode_chunk_uncached("hugs");

// --- Multithreaded batch ---------------------------------------------
std::vector<std::string_view> chunks = {"hugs", "pugs", "hug"};
auto batch = parallel_tokenize(encoder, chunks);      // vector<vector<TokenId>>

// --- Flat batch: one token array + prefix-sum offsets -----------------
FlatBatch flat = parallel_tokenize_flat(encoder, chunks);
// flat.tokens  : all token ids, chunk i occupies
//                [flat.offsets[i], flat.offsets[i + 1])
// flat.offsets : size chunks.size() + 1

// --- Persist ----------------------------------------------------------
encoder.save("model.bpe");
encoder.load("model.bpe");
```

### Python API (`fast_bpe`)

```python
import sys
import numpy as np
sys.path.append("./build")          # or pip-install / set PYTHONPATH
import fast_bpe

enc = fast_bpe.BpeEncoder()
enc.load("gpt2.bpe")

enc.encode("Hello, world!")              # -> list[int]
enc.encode_flat("Hello, world!")         # -> numpy.ndarray (uint32), zero-copy

tokens, offsets = enc.encode_batch_flat(["line 1", "line 2"])
# tokens  : numpy.ndarray (uint32)
# offsets : numpy.ndarray (uint32), length = len(chunks) + 1
# chunk i -> tokens[offsets[i]:offsets[i + 1]]

enc.decode(tokens)                       # -> str (accepts ndarray or list[int])
enc.vocab_size()                         # -> 50257
enc.merge_count()                        # -> 49992
enc.save("model.bpe")
```

| Method | Signature | Notes |
|---|---|---|
| `encode` | `(str) -> list[int]` | Pre-tokenizes, uses the per-thread cache |
| `encode_flat` | `(str) -> np.ndarray[uint32]` | Zero-copy result, GIL released during compute |
| `encode_batch` | `(Sequence[str]) -> list[list[int]]` | Persistent thread pool, GIL released |
| `encode_batch_flat` | `(Sequence[str], threads=0) -> (tokens, offsets)` | Flat zero-copy batch; `threads` is a hint, the pool size is fixed at startup |
| `decode` | `(ndarray[uint32] \| Sequence[int]) -> str` | 1-D contiguous ndarrays are read in place; GIL released |
| `load` / `save` | `(str \| PathLike) -> None` | `BPE1` binary format |
| `vocab_size` | `() -> int` | |
| `merge_count` | `() -> int` | Number of merge rules |

> The arrays returned by `encode_flat` / `encode_batch_flat` own their memory
> through a capsule, so they stay valid after the call and are freed by NumPy
> when the last reference goes away.

### Encoding pipeline

1. **Pre-tokenize** (`swar.hpp` + `pre_tokenizer.hpp`): the text is split into
   GPT-2-style pre-tokens (letter runs, digit runs, whitespace runs,
   contractions such as `'s` / `'re` / `'ll`, and punctuation runs). A trailing
   space stays attached to the following pre-token, matching GPT-2's convention
   for word-initial spaces. ASCII classification is done 8 bytes at a time with
   SWAR.
2. **Dual cursor** (`dual_cursor.hpp`): chunks longer than 256 bytes are split
   near the midpoint at a space/newline so both halves can be pre-tokenized in
   one interleaved loop without materializing an intermediate
   `vector<string_view>`.
3. **Cache lookup** (`token_cache.hpp`): pre-tokens of at most 22 bytes are
   hashed with FNV-1a and looked up in a 64-byte direct-mapped slot; hits copy
   the cached token ids straight into the output.
4. **Merge** (`encoder.cpp`): a thread-local scratchpad of `ListNode`s is built
   for the pre-token; repeatedly the lowest-rank pair is found and merged in
   place until no merge applies.
5. **Decode** (`encoder.cpp`): a flat id → bytes table is indexed directly,
   the output size is summed first, and every token is `memcpy`'d into a single
   `std::string`.

### Converting a Hugging Face tokenizer

```bash
python utils/convert_hf_to_bpe.py \
    --vocab  gpt2_tokenizer/vocab.json \
    --merges gpt2_tokenizer/merges.txt \
    --out    gpt2.bpe
```

```
[*] Loading vocab from: gpt2_tokenizer/vocab.json
[*] Total vocabulary size: 50257 tokens
[*] Parsing merge rules from: gpt2_tokenizer/merges.txt
[*] Total valid merge rules: 49992
[*] Serializing to binary file: gpt2.bpe
[+] Done! Successfully created 'gpt2.bpe' (1487.08 KB).
```

---

## Running the tests

```bash
ctest --test-dir build --output-on-failure   # via CTest (10 tests)
./build/tokenizer_tests                      # run all directly
./build/tokenizer_tests "[trainer]"          # filter by tag
./build/tokenizer_tests --list-tests
```

Available tags: `[encoder]`, `[trainer]`, `[serialization]`, `[cache]`,
`[scratchpad]`.

```
All tests passed (52 assertions in 10 test cases)
```

---

## Benchmarks

The benchmarks read `shakespeare.txt` plus `gpt2.bpe` from the repository root.

### 1. Audited 3-way benchmark (`benchmark.py`)

The primary benchmark. It splits the corpus into ~64 KiB **newline-aligned**
chunks (so no engine is penalized by mid-word or mid-indentation cuts), then
measures single- and multi-threaded encoding plus single-threaded decoding for
`fast_bpe`, OpenAI `tiktoken`, and Hugging Face `tokenizers`.

Fairness controls:

- The same chunks and GPT-2 vocabulary are used for every engine (the local
  `gpt2.bpe` for `fast_bpe`, the built-in/downloaded GPT-2 encoding for the
  others).
- Rust Rayon is pinned with `RAYON_NUM_THREADS` to the same hardware thread
  count used by `PersistentThreadPool`.
- `fast_bpe` (both boundary types) and `tiktoken` outputs are roundtrip-decoded
  and compared to the original text before timing starts.
- Both `fast_bpe` boundary types are measured: zero-copy `numpy.uint32`
  ndarrays and ordinary Python `list` objects (the latter matches the
  `tiktoken` / HF API shape).
- `tiktoken` and `tokenizers` columns are skipped automatically if those
  optional packages are not installed.

```bash
# MUST be run from the repository root (uses ./build and ./gpt2.bpe)
.venv/bin/python benchmark.py --chunk-kb 64 --warmup 3 --runs 9
```

| Flag | Default | Description |
|---|---|---|
| `--file` | `shakespeare.txt` | Input corpus |
| `--model` | `gpt2.bpe` | Binary model |
| `--chunk-kb` | `64` | Target chunk size in KiB (extended to the next newline) |
| `--warmup` | `3` | Warm-up iterations |
| `--runs` | `9` | Timed iterations (median reported) |
| `--threads` | `os.cpu_count()` | Hardware threads; also sets `RAYON_NUM_THREADS` |

### 2. C++ suite (`tokenizer_bench`)

Loads the Shakespeare corpus (~1.06 MB), trains a 350-token model, measures
serialization, then benchmarks single- and multi-threaded encoding of
`gpt2.bpe` in fixed-size 64 KiB chunks (2 warm-ups, median of 5 runs). This
suite isolates the native C++ path from the Python FFI boundary.

```bash
./build/tokenizer_bench
```

### 3. Legacy fast_bpe-only benchmark (`bench.py`)

The earlier Python script measures only `fast_bpe` (zero-copy batch encode and
ndarray decode). It remains for quick regression checks; use `benchmark.py` for
cross-library comparisons.

```bash
.venv/bin/python bench.py --chunk-kb 64 --warmup 3 --runs 7
```

---

### Benchmark results

**Hardware:** Intel Core i5-11260H @ 2.60 GHz, 6 cores / 12 threads
**Software:** Ubuntu (kernel 7.0.0-38), GCC 14.3.0, `-O3 -march=native -flto`,
CMake 4.2.3, Python 3.13.14, NumPy 2.5.4, `tiktoken 0.14.0`,
`tokenizers 0.23.2`

#### C++ suite output

```
[1/4] Loaded "shakespeare.txt" (1115394 bytes)
      Corpus Size: 1.06 MB in 100 chunks.

[2/4] Benchmarking Trainer...
      Trained to 350 tokens (94 merges) in 252.12 ms.

[3/4] Benchmarking Binary Serialization...
      Model saved to "benchmark_model.bpe" (4794 bytes) in 118.30 us.
      Model loaded (350 tokens) in 153.19 us.

[4/4] Benchmarking Inference Throughput (gpt2.bpe, 64 KiB chunks)...
      Loaded gpt2.bpe (50257 tokens, 49992 merges).
      Split into 18 chunks of ~64 KiB.
   Single-Threaded:
      Tokens Made  : 331565 tokens
      Median Speed : 41.81 MB/s
      Throughput   : 13.03 Mtok/s
      Times        : [0.0254s, 0.0251s, 0.0247s, 0.0256s, 0.0256s]

   Multi-Threaded (Hardware Concurrency = 12):
      Tokens Made  : 331565 tokens
      Median Speed : 254.88 MB/s
      Throughput   : 79.45 Mtok/s
      Times        : [0.0043s, 0.0041s, 0.0043s, 0.0042s, 0.0038s]
      Speedup      : 6.10x (median-to-median)
```

#### Audited 3-way benchmark output (`benchmark.py`)

Corpus: **1.06 MB** (1 115 394 bytes), **18 newline-aligned chunks** (~64 KB),
12 hardware threads. `fast_bpe` token counts differ from `tiktoken` / HF
because the pre-tokenizer is an independent implementation; the `fast_bpe` and
`tiktoken` outputs were verified to roundtrip to the exact source text.

```
==============================================================================================
                     AUDITED 3-WAY BPE BENCHMARK (GPT-2 VOCABULARY)
==============================================================================================
  Corpus         : shakespeare.txt (1.06 MB | 1,115,394 bytes)
  Chunks         : 18 newline-aligned chunks (~64 KB each)
  Thread Pool    : 12 hardware threads (C++ PersistentThreadPool & Rust Rayon)
==============================================================================================

==========================================================================================================
Engine             | Op     | Thr |  Median Speed |   Throughput |   Latency |    Tokens | FFI Boundary Type
----------------------------------------------------------------------------------------------------------
fast_bpe (C++20)   | Encode | 12T |    239.98 MB/s |  74.80 Mtok/s |   4.43 ms |   331,572 | Zero-Copy ndarray
fast_bpe (C++20)   | Encode | 12T |    137.40 MB/s |  42.83 Mtok/s |   7.74 ms |   331,572 | Python list[list[int]]
tiktoken (Rust)    | Encode | 12T |     45.83 MB/s |  14.56 Mtok/s |  23.21 ms |   338,025 | Python list[list[int]]
HF tokenizers      | Encode | 12T |     12.08 MB/s |   3.84 Mtok/s |  88.07 ms |   338,025 | Rust Encoding Batch
fast_bpe (C++20)   | Encode | 1T  |     34.22 MB/s |  10.67 Mtok/s |  31.08 ms |   331,572 | Python list[list[int]]
tiktoken (Rust)    | Encode | 1T  |     11.55 MB/s |   3.67 Mtok/s |  92.07 ms |   338,025 | Python list[list[int]]
HF tokenizers      | Encode | 1T  |      2.11 MB/s |   0.67 Mtok/s | 503.41 ms |   338,025 | Rust Encoding List
fast_bpe (C++20)   | Decode | 1T  |    341.01 MB/s | 106.30 Mtok/s |   3.12 ms |   331,572 | Zero-Copy ndarray
fast_bpe (C++20)   | Decode | 1T  |    269.94 MB/s |  84.14 Mtok/s |   3.94 ms |   331,572 | Python list[int]
tiktoken (Rust)    | Decode | 1T  |    117.34 MB/s |  37.29 Mtok/s |   9.07 ms |   338,025 | Python list[int]
HF tokenizers      | Decode | 1T  |     24.93 MB/s |   7.92 Mtok/s |  42.66 ms |   338,025 | Python list[int]
==========================================================================================================
```

#### Cross-library summary

| Operation | `fast_bpe` | `tiktoken` | HF `tokenizers` |
|---|---:|---:|---:|
| Encode, 12 threads | **239.98 MB/s** (ndarray) · 137.40 MB/s (list) | 45.83 MB/s | 12.08 MB/s |
| Encode, 1 thread | **34.22 MB/s** | 11.55 MB/s | 2.11 MB/s |
| Decode, 1 thread | **341.01 MB/s** (ndarray) · 269.94 MB/s (list) | 117.34 MB/s | 24.93 MB/s |

#### C++ native summary

| Phase | Metric | Result |
|---|---|---|
| Training (Shakespeare, 350-token vocab) | 94 merges | **252.12 ms** |
| Serialization | Save 4 794-byte model | **118.30 µs** |
| Serialization | Load 350-token model | **153.19 µs** |
| Encode, 1 thread (fixed 64 KiB chunks) | 331 565 tokens | **41.81 MB/s · 13.03 Mtok/s** |
| Encode, 12 threads (fixed 64 KiB chunks) | 331 565 tokens | **254.88 MB/s · 79.45 Mtok/s** |
| **Scaling** | 1 → 12 threads | **6.10×** |

**Takeaways**

- `fast_bpe` is fastest in every audited category: **5.2×** `tiktoken` and
  **19.9×** Hugging Face at 12-thread encoding; **3.0×** / **16.2×** at
  1-thread encoding.
- Decoding with a zero-copy ndarray reaches **341 MB/s**, **2.9×** `tiktoken`
  and **13.7×** HF; even the list-based decode path (**270 MB/s**) beats both
  competitors.
- The zero-copy boundary matters: ndarray encode is **1.75×** faster than
  returning `list[list[int]]` (239.98 vs 137.40 MB/s), and ndarray decode is
  **1.26×** faster than `list[int]`.
- The native C++ suite scales **6.1×** from 1 to 12 threads, confirming the
  persistent pool remains effective without Python overhead.
- `fast_bpe` emits **331 572** tokens for this corpus versus **338 025** for
  `tiktoken` / HF (~1.9 % fewer). Roundtrips are exact; the difference comes
  from the independent pre-tokenizer.

> Numbers are machine-specific and vary run-to-run (±5 %). Re-run the
> benchmarks on your own hardware for comparable figures.

---

## Binary model format (`BPE1`)

Little-endian throughout.

| Section | Fields |
|---|---|
| Header | `magic` (4 B, `BPE1`) · `version` (u8, `1`) · `vocab_size` (u32) · `merge_count` (u32) |
| Vocabulary table | repeated `vocab_size` × { `id` (u32), `len` (u32), `bytes` (len B) } |
| Merge table | repeated `merge_count` × { `left` (u32), `right` (u32), `rank` (u32), `result` (u32) } |

`load()` rejects bad magic and unsupported versions with `std::runtime_error`.
It also rebuilds the byte → token mapping and the flat decode table from the
vocabulary, so imported models (including GPT-2) roundtrip correctly.

---

## Known limitations

- **Pre-tokenization is an approximation.** The SWAR splitter mirrors the shape
  of GPT-2's split regex (letters, digits, whitespace, contractions,
  punctuation) but is not guaranteed to match it for every edge case, such as
  uncommon Unicode whitespace. Token ids and counts can therefore differ from
  `tiktoken` / Hugging Face: on `shakespeare.txt`, `fast_bpe` produces 331 572
  tokens versus 338 025 (~1.9 % fewer) while still roundtripping exactly.
- **Chunk boundaries matter for benchmarks.** `benchmark.py` aligns chunks to
  newlines to avoid bisecting words; the C++ suite and `bench.py` use fixed
  64 KiB cuts. Compare numbers produced by the same chunking strategy.
- **The pre-token cache is direct-mapped.** Colliding words simply overwrite
  each other, and words longer than 22 bytes or producing more than 8 tokens are
  never cached. This is a speed/space trade-off, not a correctness issue.
- **`threads` is advisory.** `encode_batch_flat` accepts a thread count, but the
  persistent pool is sized once at startup to `hardware_concurrency()` (caller
  included).
- The Python `decode` binding returns `str`, so byte sequences that are not valid
  UTF-8 raise an exception instead of returning surrogate-escaped bytes.
- No pre-trained model ships other than GPT-2; train your own or import one via
  `utils/convert_hf_to_bpe.py`.

---

## License

No license file is currently included. All rights reserved unless otherwise
specified by the author.
