# BPE Tokenizer

A fast **Byte Pair Encoding** tokenizer written in **C++20** (zero runtime
dependencies), with a **nanobind** Python module (`fast_bpe`), a built-in
trainer, a compact binary model format, multithreaded batch encoding, and a
benchmark suite comparing it against OpenAI `tiktoken` and Hugging Face
`tokenizers`.

---

## Features

- **Encoder** — intrusive doubly-linked-list BPE encoder; each pass applies the
  globally lowest-rank pair merge in O(n) per pass, with O(1) unlink.
- **Trainer** — frequency-weighted BPE training from raw byte corpora.
- **Parallel batch encoding** — `std::jthread` fan-out over
  `hardware_concurrency()` workers; the Python `encode_batch` binding **releases
  the GIL**.
- **Binary model format (`BPE1`)** — compact, endian-explicit, loads in
  microseconds (GPT-2: 50 257 tokens / 49 992 merges ≈ 1.5 MB).
- **HF importer** — convert Hugging Face `vocab.json` + `merges.txt` into the
  `.bpe` binary.
- **Tests** — Catch2 v3 suite (8 cases, 41 assertions) wired into CTest.

---

## Repository layout

```
BPE_Tokenizer/
├── CMakeLists.txt              # Build system (C++20, nanobind, Catch2, CTest)
├── main.cpp                    # Demo executable -> build/tokenizer_cli
├── bench.py                    # Python cross-library benchmark
├── bench/benchmark.cpp         # C++ benchmark suite -> build/tokenizer_bench
├── include/tokenizer/
│   ├── types.hpp               # TokenId, TokenPair, hash
│   ├── encoder.hpp             # BpeEncoder
│   ├── trainer.hpp             # BpeTrainer / WordItem
│   └── parallel.hpp            # parallel_tokenize()
├── src/
│   ├── encoder.cpp             # encode / decode / save / load
│   └── trainer.cpp             # training loop
├── utils/
│   ├── bindings.cpp            # nanobind module "fast_bpe"
│   └── convert_hf_to_bpe.py    # HF tokenizer -> gpt2.bpe converter
├── tests/
│   ├── test_encoder.cpp        # [encoder] [serialization]
│   └── test_trainer.cpp        # [trainer]
├── gpt2_tokenizer/             # Upstream GPT-2 vocab.json + merges.txt
└── gpt2.bpe                    # Pre-converted binary model (50 257 tokens)
```

---

## Requirements

| Component | Version |
|---|---|
| C++ compiler | C++20 (Clang, GCC, MSVC) |
| CMake | ≥ 3.20 |
| Python | ≥ 3.10 (with development headers) |
| Python packages | `nanobind` (build), `tiktoken` + `tokenizers` (benchmarks only) |
| Network | First configure (Catch2 is fetched via `FetchContent`); first benchmark run (downloads GPT-2 ranks / HF tokenizer) |

```bash
python -m venv .venv
source .venv/bin/activate
pip install nanobind tiktoken tokenizers
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

> **Note:** Clang builds get `-O3 -march=native` unconditionally. With GCC, pass
> `-DCMAKE_BUILD_TYPE=Release` to get optimizations.

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

// --- Multithreaded batch ---------------------------------------------
std::vector<std::string_view> chunks = {"hugs", "pugs", "hug"};
auto batch = parallel_tokenize(encoder, chunks);

// --- Persist ----------------------------------------------------------
encoder.save("model.bpe");
encoder.load("model.bpe");
```

Run the demo:

```bash
./build/tokenizer_cli
```

```
Text: "hugs" -> Tokens: 258 | Decoded: "hugs"
Text: "pugs" -> Tokens: 259 115 | Decoded: "pugs"
Text: "hug"  -> Tokens: 257 | Decoded: "hug"
```

### Python API (`fast_bpe`)

```python
import sys
sys.path.append("./build")          # or pip-install / set PYTHONPATH
import fast_bpe

enc = fast_bpe.BpeEncoder()
enc.load("gpt2.bpe")

enc.encode("Hello, world!")         # -> list[int]
enc.encode_batch(["line 1", "line 2"])  # -> list[list[int]], GIL released, multi-threaded
enc.decode([15496, 11])             # -> str
enc.vocab_size()                    # -> 50257
enc.save("model.bpe")
```

| Method | Signature | Notes |
|---|---|---|
| `encode` | `(str) -> list[int]` | Single chunk, single thread |
| `encode_batch` | `(Sequence[str]) -> list[list[int]]` | Parallel, releases the GIL |
| `decode` | `(Sequence[int]) -> str` | UTF-8 decodes the resulting bytes |
| `load` / `save` | `(str \| PathLike) -> None` | `BPE1` binary format |
| `vocab_size` | `() -> int` | |

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
ctest --test-dir build --output-on-failure   # via CTest (8 tests)
./build/tokenizer_tests                      # run all directly
./build/tokenizer_tests "[trainer]"          # filter by tag: [encoder] [trainer] [serialization]
./build/tokenizer_tests --list-tests
```

```
All tests passed (41 assertions in 8 test cases)
```

---

## Benchmarks

Two independent benchmarks ship with the project.

### 1. C++ suite (`tokenizer_bench`)

Measures trainer speed, binary serialization, and single- vs multi-threaded
encoding throughput on a synthetic 1.5 M-word / 10 MB corpus (seed `42`).

```bash
./build/tokenizer_bench
```

### 2. Python cross-library comparison (`bench.py`)

Compares `fast_bpe` against `tiktoken` and Hugging Face `tokenizers` on the same
5.72 MB corpus, single-threaded and batched.

```bash
# MUST be run from the repository root (uses ./build and ./gpt2.bpe)
.venv/bin/python bench.py
```

---

### Benchmark results

**Hardware:** Intel Core i5-11260H @ 2.60 GHz, 6 cores / 12 threads
**Software:** Ubuntu (kernel 7.0.0-34), clang++ 21.1.8, `-O3 -march=native`,
CMake 4.2.3 + Ninja, Python 3.13.14 (`tiktoken 0.14.0`, `tokenizers 0.23.2`)

#### C++ suite output

```
========================================================
          BPE TOKENIZER C++20 BENCHMARK SUITE
========================================================

[1/4] Generating synthetic corpus (1500000 words)...
      Corpus Size: 10.09 MB in 10000 chunks.

[2/4] Benchmarking Trainer...
      Trained to 307 tokens (51 merges) in 0.08 ms.

[3/4] Benchmarking Binary Serialization...
      Model saved to "benchmark_model.bpe" (3816 bytes) in 167.29 us.
      Model loaded (307 tokens) in 67.94 us.

[4/4] Benchmarking Inference Throughput...
   Single-Threaded:
      Time Elapsed : 21.36 s
      Tokens Made  : 6150327 tokens
      Throughput   : 0.47 MB/s (287921 tokens/sec)

   Multi-Threaded (Hardware Concurrency = 12):
      Time Elapsed : 3.88 s
      Throughput   : 2.60 MB/s (1584014 tokens/sec)
      Speedup      : 5.50x
========================================================
```

#### C++ summary

| Phase | Metric | Result |
|---|---|---|
| Training | 51 merges → 307-token vocab | **0.08 ms** |
| Serialization | Save 3 816-byte model | **167 µs** |
| Serialization | Load 307-token model | **68 µs** |
| Inference (1 thread) | 10.09 MB / 6.15 M tokens | **0.47 MB/s · 287 921 tok/s** |
| Inference (12 threads) | 10.09 MB | **2.60 MB/s · 1 584 014 tok/s** |
| **Speedup** | multi vs single | **5.50×** |

#### Python cross-library comparison

Corpus: **5.72 MB**, 5 000 chunks, 12 threads for the batch test.

```
--- [1] Single-Threaded Evaluation ---
Custom C++20 Tokenizer : 0.839 s | 6.82 MB/s
OpenAI tiktoken        : 0.276 s | 20.77 MB/s
Hugging Face Tokenizers: 1.014 s | 5.64 MB/s

--- [2] Multi-Threaded Batch Evaluation (12 threads) ---
Custom C++20 (Batch)    : 0.180 s | 31.81 MB/s
OpenAI tiktoken (Batch) : 0.151 s | 37.89 MB/s
Hugging Face (Batch)    : 0.237 s | 24.18 MB/s
```

#### Comparison summary

| Engine | Single-thread | Batched (12 threads) |
|---|---:|---:|
| **fast_bpe (this project)** | 6.82 MB/s | **31.81 MB/s** |
| OpenAI `tiktoken` | **20.77 MB/s** | **37.89 MB/s** |
| Hugging Face `tokenizers` | 5.64 MB/s | 24.18 MB/s |

**Takeaways**

- Batched, `fast_bpe` beats Hugging Face `tokenizers` by ~31 % and closes most of
  the gap to `tiktoken` (31.8 vs 37.9 MB/s).
- Single-threaded, `tiktoken` remains ~3× faster; the C++ encoder is
  competitive with, and slightly ahead of, Hugging Face's Rust implementation.
- Scaling from 1 → 12 threads gives `fast_bpe` a **4.7×** throughput gain
  (6.82 → 31.81 MB/s).

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

---

## Known limitations

- **GPT-2 ID-space mismatch.** `BpeEncoder` assumes `token_id == raw byte value`
  for the 256 base tokens, but GPT-2's `vocab.json` assigns those ids
  arbitrarily (`"!"` = 0, `"H"` = 39, …). Encoding with the shipped `gpt2.bpe`
  therefore produces different ids than `tiktoken`/HF, and `decode` of such output
  can raise `UnicodeDecodeError`. Throughput numbers above are unaffected (they
  measure speed only). Trainers trained through `BpeTrainer` are unaffected —
  they use the byte-value id space natively.
- The Python `decode` binding returns `str`, so byte sequences that are not valid
  UTF-8 raise an exception instead of returning surrogate-escaped bytes.
- No pre-trained model ships other than GPT-2; train your own or import one via
  `utils/convert_hf_to_bpe.py`.

---

## License

No license file is currently included. All rights reserved unless otherwise
specified by the author.
