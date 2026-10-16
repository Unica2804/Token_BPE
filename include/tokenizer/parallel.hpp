#pragma once

#include "encoder.hpp"
#include "types.hpp"
#include "thread_pool.hpp"
#include <cstring>
#include <span>
#include <thread>
#include <vector>
#include <algorithm>
#include <atomic>

namespace tokenizer {

struct FlatBatch {
    std::vector<TokenId> tokens;
    std::vector<uint32_t> offsets;
};

inline FlatBatch parallel_tokenize_flat(
    const BpeEncoder& encoder,
    std::span<const std::string_view> text_chunks,
    unsigned int /*requested_threads */ = 0
) {
    const size_t num_chunks = text_chunks.size();
    if (num_chunks == 0) return { {}, {0}};

    // const unsigned int hw = requested_threads > 0
    //     ? requested_threads
    //     : std::max(1u, std::thread::hardware_concurrency());
    // const unsigned int num_threads = std::min<unsigned int>(hw, static_cast<unsigned int>(num_chunks));

    // Tokneize continiously in thread safe chunks
    std::vector<std::vector<TokenId>> chunk_results(num_chunks);
    PersistentThreadPool::instance().parallel_for(num_chunks, [&](size_t i) {
        chunk_results[i] = encoder.encode_chunk(text_chunks[i]);
    });
    // std::atomic<size_t> next_chunk{0};
    // {
    //     std::vector<std::jthread> workers;
    //     workers.reserve(num_threads);

    //     for (unsigned int t = 0; t < num_threads; ++t) {
    //         workers.emplace_back([&encoder, text_chunks, &chunk_results, &next_chunk, num_chunks]() {
    //             while (true) {
    //                 size_t i = next_chunk.fetch_add(1, std::memory_order_relaxed);
    //                 if (i >= num_chunks) break;
    //                 chunk_results[i] = encoder.encode_chunk(text_chunks[i]);
    //             }
    //         });
    //     }
    // }
    // prefix sum of offsets
    FlatBatch results;
    results.offsets.resize(num_chunks + 1);
    results.offsets[0] = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        results.offsets[i + 1] = results.offsets[i] + static_cast<uint32_t>(chunk_results[i].size());
    }

    const size_t total_tokens = results.offsets[num_chunks];
    results.tokens.resize(total_tokens);

    // Block level copy into contiguous flat array
    #pragma omp parallel for if(num_chunks > 16)
    for (size_t i = 0; i < num_chunks; ++i) {
        if (!chunk_results[i].empty()) {
            std::memcpy(
                results.tokens.data() + results.offsets[i],
                chunk_results[i].data(),
                chunk_results[i].size() * sizeof(TokenId)
            );
        }
    }
    return results;
}

inline std::vector<std::vector<TokenId>> parallel_tokenize(
    const BpeEncoder& encoder,
    std::span<const std::string_view> text_chunks,
    unsigned /*int requested_threads*/ = 0
) {
    const size_t num_chunks = text_chunks.size();
    std::vector<std::vector<TokenId>> result(num_chunks);
    if (num_chunks == 0) return result;

    PersistentThreadPool::instance().parallel_for(num_chunks, [&](size_t i) {
        result[i] = encoder.encode_chunk(text_chunks[i]);
    });

    // const unsigned int hw = requested_threads > 0
    //     ? requested_threads
    //     : std::max(1u, std::thread::hardware_concurrency());
    // const unsigned int num_threads = std::min<unsigned int>(hw, static_cast<unsigned int>(num_chunks));

    // // Dynamic atomic work-stealing index prevents thread stragglers on uneven chunks
    // std::atomic<size_t> next_chunk{0};

    // std::vector<std::jthread> workers;
    // workers.reserve(num_threads);

    // for (unsigned int t = 0; t < num_threads; ++t) {
    //     workers.emplace_back([&encoder, text_chunks, &result, &next_chunk, num_chunks]() {
    //         while (true) {
    //             size_t i = next_chunk.fetch_add(1, std::memory_order_relaxed);
    //             if (i >= num_chunks) break;
    //             result[i] = encoder.encode_chunk(text_chunks[i]);
    //         }
    //     });
    // }
    return result;
}

} // namespace tokenizer