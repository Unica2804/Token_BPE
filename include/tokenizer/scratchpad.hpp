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

    ThreadScratchpad() {
        nodes.reserve(1024);
        output.reserve(1024);
    }

    void reset(size_t required_nodes) {
        nodes.clear();
        nodes.resize(required_nodes);
        output.clear();
    }
};

inline thread_local ThreadScratchpad g_scratchpad;

}