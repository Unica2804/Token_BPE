#pragma once

#include <cstdint>
#include <cstring>

namespace tokenizer::swar {

inline constexpr uint64_t HIGH_BITS = 0x8080808080808080ULL;
inline constexpr uint64_t LOW_7BITS = 0x7F7F7F7F7F7F7F7FULL;

inline constexpr uint64_t repeat_byte(uint8_t b) noexcept {
    return 0x0101010101010101ULL * b;
}

// Safe branchless SWAR check: verifies all 8 bytes are in [min_val, max_val]
// without cross-lane borrow corruption.
inline bool all_bytes_in_range(uint64_t v, uint8_t min_val, uint8_t max_val) noexcept {
    // 1. Reject if any byte has its high bit set (>= 128)
    if (v & HIGH_BITS) return false;

    // 2. Setting HIGH_BITS before subtracting min_val guarantees each byte lane >= 0x80,
    //    so subtracting min_val (<= 0x7F) can NEVER borrow from an adjacent byte lane.
    //    If a byte is >= min_val, bit 7 remains 1.
    uint64_t ge_min = (v | HIGH_BITS) - repeat_byte(min_val);

    // 3. Adding (127 - max_val) to a 7-bit value can NEVER overflow into the next lane.
    //    If a byte is <= max_val, bit 7 remains 0, so ~(le_max) has bit 7 == 1.
    uint64_t le_max = ~(v + repeat_byte(127 - max_val));

    // All 8 lanes are in range iff bit 7 is 1 in both masks for all 8 lanes.
    return (ge_min & le_max & HIGH_BITS) == HIGH_BITS;
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
    return (v ^ repeat_byte(b)) == 0;
}

inline uint64_t load64(const char* ptr) noexcept {
    uint64_t val;
    std::memcpy(&val, ptr, sizeof(val));
    return val;
}

} // namespace tokenizer::swar