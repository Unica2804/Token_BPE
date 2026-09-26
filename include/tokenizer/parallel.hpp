#pragma once

#include "tokenizer/encoder.hpp"
#include "tokenizer/types.hpp"
#include <span>
#include <thread>
#include <vector>
#include <algorithm>

namespace tokenizer {

inline std::vector<std::vector<TokenId>> parallel_tokenize(
    const BpeEncoder &encoder,
    std::span<const std::string_view> text_chunks
) {
    const size_t num_chunks = text_chunks.size();
    std::vector<std::vector<TokenId>> result(num_chunks);
    
    const unsigned int hardware_threads = std::max(1u, std::thread::hardware_concurrency());
    const size_t batch_size = (num_chunks + hardware_threads - 1) / hardware_threads;

    std::vector<std::jthread> workers;
    workers.reserve(hardware_threads);

    for (size_t t=0; t < hardware_threads; ++t) {
        size_t start = t*batch_size;
        size_t end = std::min(start + batch_size, num_chunks);
        if (start >= end) break;

        workers.emplace_back([&encoder, text_chunks, &result, start, end]() {
            for (size_t i = start; i < end; ++i) {
                result[i] = encoder.encode_chunk(text_chunks[i]);
            }
        });

    }
    return result;
}
    
}