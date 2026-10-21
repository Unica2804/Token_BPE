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
- **Tests** — Catch2 v3 suite (10 cases, 52 assertions) wired into CTest.

---

## Repository layout

```
BPE_Tokenizer/
├── CMakeLists.txt              # Build system (C++20, nanobind, Catch2, CTest)
├── main.cpp                    # Demo executable -> build/tokenizer_cli
├── bench.py                    # Python zero-copy encode/decode benchmark
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
| Python packages | `nanobind` (build), `numpy` (Python benchmark and ndarray fast paths) |
| Network | First configure only (Catch2 is fetched via `FetchContent`) |

```bash
python -m venv .venv
source .venv/bin/activate
pip install nanobind numpy
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
   contractions such as `'s` / `'re` / `'ll`, and punctuation runs). ASCII
   classification is done 8 bytes at a time with SWAR.
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

Two benchmarks ship with the project and both read `shakespeare.txt` plus
`gpt2.bpe` from the repository root.

### 1. C++ suite (`tokenizer_bench`)

Loads the Shakespeare corpus (~1.06 MB), trains a 350-token model, measures
serialization, then benchmarks single- and multi-threaded encoding of
`gpt2.bpe` in 64 KiB chunks (2 warm-ups, median of 5 runs).

```bash
./build/tokenizer_bench
```

### 2. Python zero-copy benchmark (`bench.py`)

Encodes the corpus through `encode_batch_flat`, verifies the encode/decode
roundtrip from the returned NumPy array, then benchmarks the flat batch encode
and the zero-copy decode. Only `numpy` is required.

```bash
# MUST be run from the repository root (uses ./build and ./gpt2.bpe)
.venv/bin/python bench.py --chunk-kb 64 --warmup 3 --runs 7
```

| Flag | Default | Description |
|---|---|---|
| `--file` | `shakespeare.txt` | Input corpus |
| `--model` | `gpt2.bpe` | Binary model |
| `--chunk-kb` | `64` | Chunk size in KiB |
| `--warmup` | `3` | Warm-up iterations |
| `--runs` | `7` | Timed iterations (median reported) |
| `--threads` | `os.cpu_count()` | Label only; the thread pool size is fixed at startup |

---

### Benchmark results

**Hardware:** Intel Core i5-11260H @ 2.60 GHz, 6 cores / 12 threads
**Software:** Ubuntu (kernel 7.0.0-38), GCC 14.3.0, `-O3 -march=native -flto`,
CMake 4.2.3, Python 3.13.14, NumPy 2.5.4

#### C++ suite output

```
[1/4] Loaded "shakespeare.txt" (1115394 bytes)
      Corpus Size: 1.06 MB in 100 chunks.

[2/4] Benchmarking Trainer...
      Trained to 350 tokens (94 merges) in 268.79 ms.

[3/4] Benchmarking Binary Serialization...
      Model saved to "benchmark_model.bpe" (4794 bytes) in 135.82 us.
      Model loaded (350 tokens) in 163.60 us.

[4/4] Benchmarking Inference Throughput (gpt2.bpe, 64 KiB chunks)...
      Loaded gpt2.bpe (50257 tokens, 49992 merges).
      Split into 18 chunks of ~64 KiB.
   Single-Threaded:
      Tokens Made  : 331581 tokens
      Median Speed : 34.63 MB/s
      Throughput   : 10.80 Mtok/s
      Times        : [0.0309s, 0.0307s, 0.0362s, 0.0257s, 0.0267s]

   Multi-Threaded (Hardware Concurrency = 12):
      Tokens Made  : 331581 tokens
      Median Speed : 239.33 MB/s
      Throughput   : 74.60 Mtok/s
      Times        : [0.0044s, 0.0043s, 0.0041s, 0.0046s, 0.0048s]
      Speedup      : 6.91x (median-to-median)
```

#### Python zero-copy benchmark output

```
[*] Loaded corpus: 1.06 MB (18 chunks of ~64 KB)
[*] Decode roundtrip test passed: exact string recovered from NumPy array!

>>> Running: C++20 Parallel Encode (12T) [Zero-Copy ndarray]
    Median Speed :  218.17 MB/s
    Throughput   :   68.01 Mtok/s (Tokens: 331581)

>>> Running: C++20 Single-Thread Decode [Zero-Copy ndarray -> std::string]
    Median Speed :  307.87 MB/s
```

#### Summary

| Phase | Metric | Result |
|---|---|---|
| Training (Shakespeare, 350-token vocab) | 94 merges | **268.79 ms** |
| Serialization | Save 4 794-byte model | **135.82 µs** |
| Serialization | Load 350-token model | **163.60 µs** |
| Encode, 1 thread | 1.06 MB / 331 581 tokens | **34.63 MB/s · 10.80 Mtok/s** |
| Encode, 12 threads (C++, flat) | 1.06 MB | **239.33 MB/s · 74.60 Mtok/s** |
| Encode, 12 threads (Python, ndarray) | 1.06 MB | **218.17 MB/s · 68.01 Mtok/s** |
| Decode, 1 thread (Python, ndarray) | 1.06 MB | **307.87 MB/s** |
| **Speedup** | encode, multi vs single | **6.91×** |

**Takeaways**

- The persistent thread pool scales encoding **6.9×** from 1 to 12 threads,
  reaching **239 MB/s** in C++ and **218 MB/s** through the NumPy binding.
- The zero-copy binding retains ~91 % of native flat-batch throughput; the
  remaining gap is the ndarray capsule/ownership setup per call.
- Decode is faster than encode and allocation-bounded at **~308 MB/s** on a
  single thread.

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
  uncommon Unicode whitespace. Token ids can therefore differ from
  `tiktoken` on unusual inputs.
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
