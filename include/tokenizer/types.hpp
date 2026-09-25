#pragma once
#include <cstdint>
#include <limits>
#include <iostream>

namespace tokenizer {

using TokenId = uint32_t;
constexpr TokenId INVALID_TOKEN_ID = std::numeric_limits<TokenId>::max();

// TokenPair needed for token merge
struct TokenPair {
    TokenId first;
    TokenId second;
    // Generates all comparison operators for TokenPair
    auto operator<=>(const TokenPair&) const = default;
};

// Custom hash function
struct TokenPairHash {
    size_t operator()(const TokenPair& p) const noexcept {
        uint64_t key = (static_cast<uint64_t>(p.first) << 32) | static_cast<uint64_t>(p.second);
        key ^= (key >> 30);
        key *= 0xbf58476d1ce4e5b9ULL;
        key ^= (key >> 27);
        key *= 0x94d049bb133111ebULL;
        key ^= (key >> 31);
        return static_cast<size_t>(key);
    }
};
} // namespace tokenizer