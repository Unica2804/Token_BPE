#pragma once
#include "pre_tokenizer.hpp"
#include <vector>
#include "types.hpp"
#include <string_view>

namespace tokenizer {

inline size_t find_safe_split_boundary(std::string_view text) noexcept {
    if (text.size() < 256) return text.size();
    const size_t mid = text.size() / 2;

    for (size_t i = mid; i > mid - 128 && i > 0; --i) {
        if (text[i] == ' ' || text[i] == '\n') {
            return i; // Split right BEFORE the space so stream B starts with ' ' (GPT-2 style)
        }
    }

    for (size_t i = mid; i < mid + 128 && i < text.size(); ++i) {
        if (text[i] == ' ' || text[i] == '\n') {
            return i;
        }
    }
    return text.size();
}

// Fused dual-cursor execution: immediately feeds each pre-token to `emit_fn`
// without materializing intermediate std::vector<std::string_view> arrays.
template <typename EmitFn>
inline void process_dual_cursor(
    std::string_view text,
    std::vector<TokenId>& out_a,
    std::vector<TokenId>& out_b,
    EmitFn&& emit_fn
) {
    if (text.empty()) return;

    const size_t split = find_safe_split_boundary(text);

    LexerCursor cursor_a{text.data(), text.data() + split};
    LexerCursor cursor_b{text.data() + split, text.data() + text.size()};

    while (cursor_a.has_more() && cursor_b.has_more()) {
        std::string_view tok_a = next_token(cursor_a);
        emit_fn(tok_a, out_a);

        std::string_view tok_b = next_token(cursor_b);
        emit_fn(tok_b, out_b);
    }

    while (cursor_a.has_more()) {
        emit_fn(next_token(cursor_a), out_a);
    }

    while (cursor_b.has_more()) {
        emit_fn(next_token(cursor_b), out_b);
    }
}

} // namespace tokenizer