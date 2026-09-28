#include "tokenizer/types.hpp"
#include "tokenizer/encoder.hpp"
#include "tokenizer/trainer.hpp"
#include "tokenizer/parallel.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <numeric>
#include <iomanip>
#include <filesystem>

using namespace tokenizer;

// Helper: Generates pseudo-natural language text
std::vector<std::string> generate_synthetic_corpus(size_t total_words, size_t chunk_count) {
    const std::vector<std::string> dictionary = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "transformer", "attention", "token", "byte", "pair", "encoding",
        "optimization", "throughput", "rust", "cpp20", "performance", "memory"
    };

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, dictionary.size() - 1);

    std::vector<std::string> chunks(chunk_count);
    size_t words_per_chunk = total_words / chunk_count;

    for (size_t c = 0; c < chunk_count; ++c) {
        std::string chunk;
        chunk.reserve(words_per_chunk * 7);
        for (size_t w = 0; w < words_per_chunk; ++w) {
            chunk += dictionary[dist(rng)];
            chunk += " ";
        }
        chunks[c] = std::move(chunk);
    }
    return chunks;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "          BPE TOKENIZER C++20 BENCHMARK SUITE           \n";
    std::cout << "========================================================\n\n";

    // 1. Data Generation (~10MB of text)
    const size_t TOTAL_WORDS = 1'500'000;
    const size_t NUM_CHUNKS = 10'000;
    std::cout << "[1/4] Generating synthetic corpus (" << TOTAL_WORDS << " words)...\n";
    
    auto raw_chunks = generate_synthetic_corpus(TOTAL_WORDS, NUM_CHUNKS);
    size_t total_bytes = 0;
    for (const auto& ch : raw_chunks) total_bytes += ch.size();
    
    double mb_size = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
    std::cout << "      Corpus Size: " << std::fixed << std::setprecision(2) 
              << mb_size << " MB in " << NUM_CHUNKS << " chunks.\n\n";

    // 2. Training Benchmark
    std::cout << "[2/4] Benchmarking Trainer...\n";
    std::vector<WordItem> training_data = {
        {{'t','r','a','n','s','f','o','r','m','e','r'}, 5000},
        {{'a','t','t','e','n','t','i','o','n'}, 4000},
        {{'t','o','k','e','n','i','z','e','r'}, 3500},
        {{'e','n','c','o','d','i','n','g'}, 3000},
        {{'o','p','t','i','m','i','z','a','t','i','o','n'}, 2500},
        {{'p','e','r','f','o','r','m','a','n','c','e'}, 2000},
        {{'t','h','e'}, 10000},
        {{'q','u','i','c','k'}, 1200},
        {{'b','r','o','w','n'}, 1100}
    };

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

    // 4. Inference / Encoding Benchmark
    std::cout << "[4/4] Benchmarking Inference Throughput...\n";

    // Create string views for zero-copy parallel processing
    std::vector<std::string_view> view_chunks;
    view_chunks.reserve(raw_chunks.size());
    for (const auto& ch : raw_chunks) view_chunks.emplace_back(ch);

    // --- Single-Threaded Run ---
    auto st_start = std::chrono::steady_clock::now();
    size_t st_total_tokens = 0;
    for (const auto& view : view_chunks) {
        auto tokens = loaded_encoder.encode_chunk(view);
        st_total_tokens += tokens.size();
    }
    auto st_end = std::chrono::steady_clock::now();

    std::chrono::duration<double> st_sec = st_end - st_start;
    double st_mb_per_sec = mb_size / st_sec.count();
    double st_tokens_per_sec = static_cast<double>(st_total_tokens) / st_sec.count();

    std::cout << "   Single-Threaded:\n";
    std::cout << "      Time Elapsed : " << st_sec.count() << " s\n";
    std::cout << "      Tokens Made  : " << st_total_tokens << " tokens\n";
    std::cout << "      Throughput   : " << std::fixed << std::setprecision(2)
              << st_mb_per_sec << " MB/s (" 
              << static_cast<uint64_t>(st_tokens_per_sec) << " tokens/sec)\n\n";

    // --- Multi-Threaded Run (std::jthread) ---
    auto mt_start = std::chrono::steady_clock::now();
    auto mt_results = parallel_tokenize(loaded_encoder, view_chunks);
    auto mt_end = std::chrono::steady_clock::now();

    size_t mt_total_tokens = 0;
    for (const auto& vec : mt_results) mt_total_tokens += vec.size();

    std::chrono::duration<double> mt_sec = mt_end - mt_start;
    double mt_mb_per_sec = mb_size / mt_sec.count();
    double mt_tokens_per_sec = static_cast<double>(mt_total_tokens) / mt_sec.count();

    std::cout << "   Multi-Threaded (Hardware Concurrency = " 
              << std::thread::hardware_concurrency() << "):\n";
    std::cout << "      Time Elapsed : " << mt_sec.count() << " s\n";
    std::cout << "      Throughput   : " << std::fixed << std::setprecision(2)
              << mt_mb_per_sec << " MB/s (" 
              << static_cast<uint64_t>(mt_tokens_per_sec) << " tokens/sec)\n";
    std::cout << "      Speedup      : " << std::fixed << std::setprecision(2)
              << (st_sec.count() / mt_sec.count()) << "x\n";

    std::cout << "========================================================\n";
    return 0;
}