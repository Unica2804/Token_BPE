#pragma once

#include "types.hpp"
#include "scratchpad.hpp"
#include <string>
#include <array>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <filesystem>

namespace tokenizer {

struct MergeInfo {
    uint32_t rank{0};
    TokenId new_id{INVALID_TOKEN_ID};
};

class BpeEncoder {
public:
    using MergeRanks = std::unordered_map<TokenPair, MergeInfo, TokenPairHash>;

    BpeEncoder();

    void add_merge(TokenId left, TokenId right, uint32_t rank, TokenId new_token_id);
    void set_byte_token(uint8_t byte_val, TokenId token_id);

    [[nodiscard]] std::vector<TokenId> encode_chunk(std::string_view chunk) const;
    [[nodiscard]] std::vector<TokenId> encode_chunk_uncached(std::string_view chunk) const;

    void encode_chunk_to_buffer(std::string_view chunk, std::vector<TokenId>& out) const;
    void encode_chunk_uncached_to_buffer(std::string_view chunk, std::vector<TokenId>& out) const;

    [[nodiscard]] std::string decode(const std::vector<TokenId>& tokens) const;

    [[nodiscard]] size_t vocab_size() const noexcept { return id_to_bytes_.size(); }
    [[nodiscard]] size_t merge_count() const noexcept { return merge_ranks_.size(); }

    void save(const std::filesystem::path& path) const;
    void load(const std::filesystem::path& path);

private:
    MergeRanks merge_ranks_;
    std::unordered_map<TokenId, std::string> id_to_bytes_;
    std::unordered_map<std::string, TokenId> bytes_to_id_;
    std::array<TokenId, 256> byte_to_token_id{};
};

} // namespace tokenizer