#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>
#include <bit>

namespace tokenizer::swar {

// Bit mask high bits
inline constexpr uint64_t HIGH_BITS = 0x8080808080808080ULL;

//Broadcasting helper
inline constexpr uint64_t repeat_byte(uint8_t b) {
    return 0x0101010101010101ULL * b;
}

inline bool all_bytes_in_range(uint64_t v, uint8_t min_val, uint8_t max_val) noexcept {
    uint64_t diff = repeat_byte(max_val - min_val);
    uint64_t offset = v - repeat_byte(min_val);
    return (((offset | (diff - offset)) & HIGH_BITS) == 0);
}

inline bool all_ascii_lower(uint64_t v) noexcept {
    return all_bytes_in_range(v, 'a', 'z');

}

inline bool all_ascii_upper(uint64_t v) noexcept {
    return all_bytes_in_range(v, 'A', 'Z');
}

inline bool all_ascii_letters(uint64_t v) noexcept {
    uint64_t lower_check = (v | repeat_byte(0x20));
    return all_ascii_lower(lower_check);
}

inline bool all_ascii_digits(uint64_t v) noexcept {
    return all_bytes_in_range(v, '0', '9');
}

inline bool all_bytes_equal(uint64_t v, uint8_t b) noexcept {
    uint64_t xor_diff = v ^ repeat_byte(b);
    return xor_diff == 0;
}

inline uint64_t load64(const char* ptr) noexcept {
    uint64_t val;
    std::memcpy(&val, ptr, sizeof(val));
    return val;
}

}