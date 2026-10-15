#pragma once

#include "encoder.hpp"
#include "types.hpp"
#include <span>
#include <thread>
#include <vector>
#include <algorithm>
#include <atomic>

namespace tokenizer {

inline std::vector<std::vector<TokenId>> parallel_tokenize(
    const BpeEncoder& encoder,
    std::span<const std::string_view> text_chunks,
    unsigned int requested_threads = 0
) {
    const size_t num_chunks = text_chunks.size();
    std::vector<std::vector<TokenId>> result(num_chunks);
    if (num_chunks == 0) return result;

    const unsigned int hw = requested_threads > 0
        ? requested_threads
        : std::max(1u, std::thread::hardware_concurrency());
    const unsigned int num_threads = std::min<unsigned int>(hw, static_cast<unsigned int>(num_chunks));

    // Dynamic atomic work-stealing index prevents thread stragglers on uneven chunks
    std::atomic<size_t> next_chunk{0};

    std::vector<std::jthread> workers;
    workers.reserve(num_threads);

    for (unsigned int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&encoder, text_chunks, &result, &next_chunk, num_chunks]() {
            while (true) {
                size_t i = next_chunk.fetch_add(1, std::memory_order_relaxed);
                if (i >= num_chunks) break;
                result[i] = encoder.encode_chunk(text_chunks[i]);
            }
        });
    }
    return result;
}

} // namespace tokenizer