#pragma once

#include "types.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <span>

namespace tokenizer {

class TokenCache {
public:
    // Power-of-two slot count for 1-cycle bitmask indexing (8192 * 64B = 512 KB in L2/L3 cache)
    static constexpr size_t NUM_SLOTS = 8192;
    static constexpr size_t SLOT_MASK = NUM_SLOTS - 1;
    static constexpr size_t MAX_CACHABLE_LENGTH = 22;
    static constexpr size_t MAX_CACHED_TOKENS = 8;

    // Exactly 64 bytes (1 CPU cache line): zero heap pointers!
    struct alignas(64) CacheSlot {
        uint64_t hash{0};
        uint8_t word_len{0};
        uint8_t token_count{0};
        char word[MAX_CACHABLE_LENGTH]{};
        TokenId tokens[MAX_CACHED_TOKENS]{};
    };
    static_assert(sizeof(CacheSlot) == 64, "CacheSlot must fit in a single 64B cache line");

    TokenCache() = default;

    // Fast 64-bit FNV-1a hash
    [[nodiscard]] static constexpr uint64_t hash_word(std::string_view sv) noexcept {
        uint64_t h = 14695981039346656037ULL;
        for (char c : sv) {
            h ^= static_cast<uint8_t>(c);
            h *= 1099511628211ULL;
        }
        return h == 0 ? 1ULL : h; // Reserve 0 for empty slot
    }

    // Lookup returns a zero-copy span into the inline slot if hit, or empty span on miss
    [[nodiscard]] std::span<const TokenId> get(std::string_view word, uint64_t h) const noexcept {
        const CacheSlot& slot = slots_[h & SLOT_MASK];
        if (slot.hash == h && slot.word_len == word.size()) {
            if (std::memcmp(slot.word, word.data(), word.size()) == 0) {
                return std::span<const TokenId>(slot.tokens, slot.token_count);
            }
        }
        return {};
    }

    // Insert directly into the flat slot (overwrites older collision without clearing whole table)
    void put(std::string_view word, uint64_t h, const TokenId* tok_ptr, size_t tok_count) noexcept {
        if (word.size() > MAX_CACHABLE_LENGTH || tok_count == 0 || tok_count > MAX_CACHED_TOKENS) {
            return;
        }
        CacheSlot& slot = slots_[h & SLOT_MASK];
        slot.hash = h;
        slot.word_len = static_cast<uint8_t>(word.size());
        slot.token_count = static_cast<uint8_t>(tok_count);
        std::memcpy(slot.word, word.data(), word.size());
        std::memcpy(slot.tokens, tok_ptr, tok_count * sizeof(TokenId));
    }

    void clear() noexcept {
        for (auto& slot : slots_) {
            slot.hash = 0;
        }
    }

private:
    std::array<CacheSlot, NUM_SLOTS> slots_{};
};

} // namespace tokenizer