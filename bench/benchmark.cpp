#include "tokenizer/types.hpp"
#include "tokenizer/encoder.hpp"
#include "tokenizer/trainer.hpp"
#include "tokenizer/parallel.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <numeric>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace tokenizer;

int main() {
    std::cout << "========================================================\n";
    std::cout << "          BPE TOKENIZER C++20 BENCHMARK SUITE           \n";
    std::cout << "========================================================\n\n";

    // 1. Load shakespeare.txt
    const std::filesystem::path corpus_path = "shakespeare.txt";
    std::ifstream file(corpus_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << corpus_path << "\n";
        return 1;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    std::string corpus = ss.str();

    const size_t NUM_CHUNKS = 100;
    const size_t chunk_size = (corpus.size() + NUM_CHUNKS - 1) / NUM_CHUNKS;
    std::vector<std::string> raw_chunks;
    raw_chunks.reserve(NUM_CHUNKS);
    for (size_t i = 0; i < corpus.size(); i += chunk_size) {
        raw_chunks.push_back(corpus.substr(i, chunk_size));
    }

    size_t total_bytes = corpus.size();
    double mb_size = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
    std::cout << "[1/4] Loaded " << corpus_path << " (" << total_bytes << " bytes)\n";
    std::cout << "      Corpus Size: " << std::fixed << std::setprecision(2)
              << mb_size << " MB in " << raw_chunks.size() << " chunks.\n\n";

    // 2. Training Benchmark (train on shakespeare.txt word frequencies)
    std::cout << "[2/4] Benchmarking Trainer...\n";
    std::unordered_map<std::string, uint32_t> word_counts;
    std::istringstream iss(corpus);
    std::string word;
    while (iss >> word) {
        ++word_counts[word];
    }
    std::vector<WordItem> training_data;
    training_data.reserve(word_counts.size());
    for (const auto& [w, freq] : word_counts) {
        WordItem item;
        item.frequency = freq;
        item.tokens.reserve(w.size());
        for (char c : w) item.tokens.push_back(static_cast<unsigned char>(c));
        training_data.push_back(std::move(item));
    }

    BpeEncoder encoder;
    const size_t TARGET_VOCAB = 350; // 256 base bytes + 94 merges

    auto t_start = std::chrono::steady_clock::now();
    BpeTrainer::train(training_data, TARGET_VOCAB, encoder);
    auto t_end = std::chrono::steady_clock::now();
    
    std::chrono::duration<double, std::milli> train_ms = t_end - t_start;
    std::cout << "      Trained to " << encoder.vocab_size() 
              << " tokens (" << encoder.merge_count() << " merges) in "
              << train_ms.count() << " ms.\n\n";

    // 3. Serialization Benchmark
    std::cout << "[3/4] Benchmarking Binary Serialization...\n";
    const std::filesystem::path model_path = "benchmark_model.bpe";

    auto save_start = std::chrono::steady_clock::now();
    encoder.save(model_path);
    auto save_end = std::chrono::steady_clock::now();
    
    std::chrono::duration<double, std::micro> save_us = save_end - save_start;
    auto file_bytes = std::filesystem::file_size(model_path);
    std::cout << "      Model saved to " << model_path << " (" << file_bytes << " bytes) in "
              << save_us.count() << " us.\n";

    BpeEncoder loaded_encoder;
    auto load_start = std::chrono::steady_clock::now();
    loaded_encoder.load(model_path);
    auto load_end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::micro> load_us = load_end - load_start;
    std::cout << "      Model loaded (" << loaded_encoder.vocab_size() << " tokens) in "
              << load_us.count() << " us.\n\n";

    // Clean up temporary model file
    std::filesystem::remove(model_path);

    // 4. Inference / Encoding Benchmark.
    // Fair comparison with bench.py defaults: gpt2.bpe model, 64 KiB chunks,
    // 2 warm-up runs, median of 5 timed runs (single-threaded + parallel).
    std::cout << "[4/4] Benchmarking Inference Throughput (gpt2.bpe, 64 KiB chunks)...\n";

    const std::filesystem::path gpt2_path = "gpt2.bpe";
    BpeEncoder gpt2_encoder;
    try {
        gpt2_encoder.load(gpt2_path);
    } catch (const std::exception& e) {
        std::cerr << "      Cannot load " << gpt2_path << ": " << e.what() << "\n";
        return 1;
    }
    std::cout << "      Loaded " << gpt2_path << " (" << gpt2_encoder.vocab_size()
              << " tokens, " << gpt2_encoder.merge_count() << " merges).\n";

    constexpr size_t FAIR_CHUNK_SIZE = 64 * 1024; // must match bench.py --chunk-kb default
    constexpr int WARMUP_RUNS = 2;               // must match bench.py --warmup default
    constexpr int TIMED_RUNS = 5;                // must match bench.py --runs default

    std::vector<std::string> fair_raw_chunks;
    fair_raw_chunks.reserve((corpus.size() + FAIR_CHUNK_SIZE - 1) / FAIR_CHUNK_SIZE);
    for (size_t i = 0; i < corpus.size(); i += FAIR_CHUNK_SIZE) {
        fair_raw_chunks.push_back(corpus.substr(i, std::min(FAIR_CHUNK_SIZE, corpus.size() - i)));
    }
    std::vector<std::string_view> fair_views;
    fair_views.reserve(fair_raw_chunks.size());
    for (const auto& ch : fair_raw_chunks) fair_views.emplace_back(ch);
    std::cout << "      Split into " << fair_views.size() << " chunks of ~64 KiB.\n";

    auto median_of = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto print_times = [](const std::vector<double>& durs) {
        std::cout << "      Times        : [";
        for (size_t i = 0; i < durs.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << durs[i] << "s";
            if (i + 1 < durs.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    };

    // --- Single-Threaded: warm-up, then median of TIMED_RUNS ---
    for (int w = 0; w < WARMUP_RUNS; ++w) {
        volatile size_t sink = 0;
        for (const auto& view : fair_views) sink += gpt2_encoder.encode_chunk(view).size();
        (void)sink;
    }
    std::vector<double> st_durs;
    st_durs.reserve(TIMED_RUNS);
    size_t st_total_tokens = 0;
    for (int r = 0; r < TIMED_RUNS; ++r) {
        auto st_start = std::chrono::steady_clock::now();
        size_t run_tokens = 0;
        for (const auto& view : fair_views) {
            run_tokens += gpt2_encoder.encode_chunk(view).size();
        }
        std::chrono::duration<double> st_sec = std::chrono::steady_clock::now() - st_start;
        st_durs.push_back(st_sec.count());
        if (r == 0) st_total_tokens = run_tokens;
    }
    double st_med = median_of(st_durs);

    std::cout << "   Single-Threaded:\n";
    std::cout << "      Tokens Made  : " << st_total_tokens << " tokens\n";
    std::cout << "      Median Speed : " << std::fixed << std::setprecision(2)
              << (mb_size / st_med) << " MB/s\n";
    std::cout << "      Throughput   : " << std::fixed << std::setprecision(2)
              << (static_cast<double>(st_total_tokens) / st_med / 1e6) << " Mtok/s\n";
    print_times(st_durs);
    std::cout << "\n";

    // --- Multi-Threaded (std::jthread): warm-up, then median of TIMED_RUNS ---
    for (int w = 0; w < WARMUP_RUNS; ++w) {
        auto warm = parallel_tokenize(gpt2_encoder, fair_views);
        volatile size_t sink = 0;
        for (const auto& vec : warm) sink += vec.size();
        (void)sink;
    }
    std::vector<double> mt_durs;
    mt_durs.reserve(TIMED_RUNS);
    size_t mt_total_tokens = 0;
    for (int r = 0; r < TIMED_RUNS; ++r) {
        auto mt_start = std::chrono::steady_clock::now();
        auto mt_results = parallel_tokenize(gpt2_encoder, fair_views);
        std::chrono::duration<double> mt_sec = std::chrono::steady_clock::now() - mt_start;
        mt_durs.push_back(mt_sec.count());
        if (r == 0) {
            for (const auto& vec : mt_results) mt_total_tokens += vec.size();
        }
    }
    double mt_med = median_of(mt_durs);

    std::cout << "   Multi-Threaded (Hardware Concurrency = "
              << std::thread::hardware_concurrency() << "):\n";
    std::cout << "      Tokens Made  : " << mt_total_tokens << " tokens\n";
    std::cout << "      Median Speed : " << std::fixed << std::setprecision(2)
              << (mb_size / mt_med) << " MB/s\n";
    std::cout << "      Throughput   : " << std::fixed << std::setprecision(2)
              << (static_cast<double>(mt_total_tokens) / mt_med / 1e6) << " Mtok/s\n";
    print_times(mt_durs);
    std::cout << "      Speedup      : " << std::fixed << std::setprecision(2)
              << (st_med / mt_med) << "x (median-to-median)\n";

    std::cout << "========================================================\n";
    return 0;
}