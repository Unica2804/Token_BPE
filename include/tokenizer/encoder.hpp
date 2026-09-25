#pragma once

#include <cstdint>
#include "types.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <iostream>

namespace tokenizer {
    struct ListNode {
    TokenId token{INVALID_TOKEN_ID};
    int32_t prev{-1};
    int32_t next{-1};  
};

class BpeEncoder {
    public:
        using MergeRanks = std::unordered_map<TokenPair, uint32_t, TokenPairHash>;
        BpeEncoder();
        void add_merge(TokenId left, TokenId right, uint32_t rank, TokenId new_token_id);
        [[nodiscard]]std::vector<TokenId> encode_chunk(std::string_view chunk) const;
        [[nodiscard]]std::string decode(const std::vector<TokenId>& tokens) const;

        [[nodiscard]] size_t vocab_size() const noexcept { return id_to_bytes_.size(); }
    
    private:
        MergeRanks merge_ranks_;
        std::unordered_map<TokenId, std::string> id_to_bytes_;
        std::unordered_map<std::string, TokenId> bytes_to_id_;
};
} // namespace tokenizer