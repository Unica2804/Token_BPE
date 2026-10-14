#pragma once
#include "pre_tokenizer.hpp"
#include <vector>
#include <string_view>

namespace tokenizer {

inline size_t find_safe_split_boundary(std::string_view text) noexcept {
    if(text.size() < 256) return text.size();
    size_t mid = text.size() / 2;

    for(size_t i = mid; i>mid-128 && i>0; --i) {
        if(text[i] == ' ' || text[i] == '\n') {
            return i+1;
        }
    }

    for(size_t i = mid; i<mid+128 && i<text.size(); ++i) {
        if(text[i] == ' ' || text[i] == '\n') {
            return i+1;
        }
    }
    return text.size();
}

inline void pre_tokenize_dual_cursor(
    std::string_view text,
    std::vector<std::string_view> & out_slice_a,
    std::vector<std::string_view> & out_slice_b
) {
    if (text.empty()) return;

    size_t split = find_safe_split_boundary(text);

    LexerCursor cursor_a{text.data(), text.data() + split};
    LexerCursor cursor_b{text.data() + split, text.data() + text.size()};

    out_slice_a.reserve(split/3);
    out_slice_b.reserve((text.size()-split)/3);

    while(cursor_a.has_more() && cursor_b.has_more()) {
        // std::string_view token_a = next_token(currsor_a);
        // std::string_view token_b = next_token(currsor_b);

        out_slice_a.push_back(next_token(cursor_a));
        out_slice_b.push_back(next_token(cursor_b));

    }


    while(cursor_a.has_more()) {
        out_slice_a.push_back(next_token(cursor_a));
    }

    while(cursor_b.has_more()) {
        out_slice_b.push_back(next_token(cursor_b));
    }
}

}