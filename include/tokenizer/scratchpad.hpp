#pragma once

#include "types.hpp"
#include <vector>

namespace tokenizer {

struct ListNode {
    TokenId token{INVALID_TOKEN_ID};
    int32_t prev{-1};
    int32_t next{-1};
};

struct ThreadScratchpad {
    std::vector<ListNode> nodes;
    std::vector<TokenId> output;

    static constexpr size_t MAX_SCRATCHPAD_CAPACITY = 262144;

    ThreadScratchpad() {
        nodes.reserve(4096);
        output.reserve(4096);
    }

    void reset(size_t required_nodes) {
        if (required_nodes > MAX_SCRATCHPAD_CAPACITY) {
            nodes.clear();
            nodes.shrink_to_fit();
            nodes.resize(required_nodes);
        }
        else {
            nodes.clear();
            nodes.resize(required_nodes);
        }
        output.clear();
    }
};

inline thread_local ThreadScratchpad g_scratchpad;

}