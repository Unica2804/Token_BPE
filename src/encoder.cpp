#include "tokenizer/encoder.hpp"
#include "tokenizer/token_cache.hpp"
#include "tokenizer/dual_cursor.hpp"
#include "tokenizer/types.hpp"
#include <fstream>
#include <stdexcept>
#include <array>
#include <limits>
#include <string_view>
#include <vector>

namespace tokenizer {

namespace {
    constexpr std::array<char, 4> MAGIC = {'B', 'P', 'E', '1'};
    constexpr uint8_t FORMAT_VERSION = 1;
}

static thread_local TokenCache tl_cache;
static thread_local std::vector<TokenId> tl_tokens_b;

BpeEncoder::BpeEncoder() {
    merge_ranks_.reserve(65536);
    id_to_bytes_.reserve(65536);
    bytes_to_id_.reserve(65536);

    for (uint32_t i = 0; i < 256; ++i) {
        std::string byte_str(1, static_cast<char>(i));
        id_to_bytes_[i] = byte_str;
        bytes_to_id_[byte_str] = i;
        byte_to_token_id[i] = i; // Fixed: initialize default byte mapping!
    }
}

void BpeEncoder::set_byte_token(uint8_t byte_val, TokenId token_id) {
    byte_to_token_id[byte_val] = token_id;
}

void BpeEncoder::add_merge(TokenId left, TokenId right, uint32_t rank, TokenId new_token_id) {
    merge_ranks_[{left, right}] = MergeInfo{rank, new_token_id};
    std::string merged_bytes = id_to_bytes_[left] + id_to_bytes_[right];
    id_to_bytes_[new_token_id] = merged_bytes;
    bytes_to_id_[merged_bytes] = new_token_id;
}

void BpeEncoder::encode_chunk_to_buffer(std::string_view word, std::vector<TokenId>& out) const {
    if (word.empty()) return;

    // Single-byte fast path: 1 byte NEVER merges with anything and needs no cache lookup
    if (word.size() == 1) {
        out.push_back(byte_to_token_id[static_cast<uint8_t>(word[0])]);
        return;
    }

    uint64_t word_hash = 0;
    if (word.size() <= TokenCache::MAX_CACHABLE_LENGTH) {
        word_hash = TokenCache::hash_word(word);
        auto cached = tl_cache.get(word, word_hash);
        if (!cached.empty()) {
            out.insert(out.end(), cached.begin(), cached.end());
            return;
        }
    }

    const size_t start_offset = out.size();
    encode_chunk_uncached_to_buffer(word, out);

    if (word.size() <= TokenCache::MAX_CACHABLE_LENGTH) {
        const size_t produced = out.size() - start_offset;
        tl_cache.put(word, word_hash, out.data() + start_offset, produced);
    }
}

void BpeEncoder::encode_chunk_uncached_to_buffer(std::string_view chunk, std::vector<TokenId>& out) const {
    if (chunk.empty()) return;

    const size_t n = chunk.size();
    if (n == 1) {
        out.push_back(byte_to_token_id[static_cast<uint8_t>(chunk[0])]);
        return;
    }

    g_scratchpad.reset(n);
    auto& nodes = g_scratchpad.nodes;

    for (size_t i = 0; i < n; ++i) {
        uint8_t b = static_cast<uint8_t>(chunk[i]);
        nodes[i].token = byte_to_token_id[b];
        nodes[i].prev = static_cast<int32_t>(i) - 1;
        nodes[i].next = static_cast<int32_t>(i) + 1;
    }
    nodes.back().next = -1;

    while (true) {
        uint32_t lowest_rank = std::numeric_limits<uint32_t>::max();
        int32_t best_node_index = -1;
        TokenId best_new_id = INVALID_TOKEN_ID;

        int32_t curr = 0;
        while (curr != -1 && nodes[curr].next != -1) {
            int32_t nxt = nodes[curr].next;
            TokenPair pair{nodes[curr].token, nodes[nxt].token};
            if (auto it = merge_ranks_.find(pair); it != merge_ranks_.end()) {
                if (it->second.rank < lowest_rank) {
                    lowest_rank = it->second.rank;
                    best_node_index = curr;
                    best_new_id = it->second.new_id;
                }
            }
            curr = nxt;
        }

        if (best_node_index == -1) break;

        // O(1) in-place merge with ZERO string allocations or extra hash lookups
        int32_t merge_left = best_node_index;
        int32_t merge_right = nodes[merge_left].next;
        int32_t right_neighbour = nodes[merge_right].next;

        nodes[merge_left].token = best_new_id;
        nodes[merge_left].next = right_neighbour;
        if (right_neighbour != -1) {
            nodes[right_neighbour].prev = merge_left;
        }
    }

    int32_t curr = 0;
    while (curr != -1) {
        out.push_back(nodes[curr].token);
        curr = nodes[curr].next;
    }
}

std::vector<TokenId> BpeEncoder::encode_chunk(std::string_view chunk) const {
    if (chunk.empty()) return {};

    std::vector<TokenId> tokens_a;
    tokens_a.reserve(chunk.size() / 3);

    tl_tokens_b.clear();
    if (tl_tokens_b.capacity() < chunk.size() / 6) {
        tl_tokens_b.reserve(chunk.size() / 6);
    }

    process_dual_cursor(
        chunk,
        tokens_a,
        tl_tokens_b,
        [this](std::string_view word, std::vector<TokenId>& out) {
            this->encode_chunk_to_buffer(word, out);
        }
    );

    tokens_a.insert(tokens_a.end(), tl_tokens_b.begin(), tl_tokens_b.end());
    return tokens_a;
}

std::vector<TokenId> BpeEncoder::encode_chunk_uncached(std::string_view chunk) const {
    std::vector<TokenId> result;
    encode_chunk_uncached_to_buffer(chunk, result);
    return result;
}

std::string BpeEncoder::decode(const std::vector<TokenId>& tokens) const {
    std::string result;
    result.reserve(tokens.size() * 4);
    for (TokenId token : tokens) {
        if (auto it = id_to_bytes_.find(token); it != id_to_bytes_.end()) {
            result += it->second;
        }
    }
    return result;
}

void BpeEncoder::save(const std::filesystem::path& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath.string());
    }

    out.write(MAGIC.data(), MAGIC.size());
    out.write(reinterpret_cast<const char*>(&FORMAT_VERSION), sizeof(FORMAT_VERSION));

    uint32_t v_size = static_cast<uint32_t>(id_to_bytes_.size());
    uint32_t m_size = static_cast<uint32_t>(merge_ranks_.size());
    out.write(reinterpret_cast<const char*>(&v_size), sizeof(v_size));
    out.write(reinterpret_cast<const char*>(&m_size), sizeof(m_size));

    for (const auto& [id, bytes] : id_to_bytes_) {
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        uint32_t bytes_size = static_cast<uint32_t>(bytes.size());
        out.write(reinterpret_cast<const char*>(&bytes_size), sizeof(bytes_size));
        out.write(bytes.data(), bytes_size);
    }

    for (const auto& [pair, info] : merge_ranks_) {
        out.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
        out.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
        out.write(reinterpret_cast<const char*>(&info.rank), sizeof(info.rank));
        out.write(reinterpret_cast<const char*>(&info.new_id), sizeof(info.new_id));
    }
}

void BpeEncoder::load(const std::filesystem::path& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath.string());
    }

    std::array<char, 4> file_magic{};
    in.read(file_magic.data(), file_magic.size());
    if (file_magic != MAGIC) {
        throw std::runtime_error("Invalid file format: " + filepath.string());
    }

    uint8_t version = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != FORMAT_VERSION) {
        throw std::runtime_error("Unsupported format version: " + std::to_string(version));
    }

    uint32_t v_size = 0;
    uint32_t m_size = 0;
    in.read(reinterpret_cast<char*>(&v_size), sizeof(v_size));
    in.read(reinterpret_cast<char*>(&m_size), sizeof(m_size));

    id_to_bytes_.clear();
    bytes_to_id_.clear();
    merge_ranks_.clear();
    tl_cache.clear();

    id_to_bytes_.reserve(v_size);
    bytes_to_id_.reserve(v_size);
    merge_ranks_.reserve(m_size);

    for (uint32_t i = 0; i < v_size; ++i) {
        TokenId id = 0;
        uint32_t len = 0;
        in.read(reinterpret_cast<char*>(&id), sizeof(id));
        in.read(reinterpret_cast<char*>(&len), sizeof(len));

        std::string bytes(len, '\0');
        in.read(bytes.data(), len);

        id_to_bytes_[id] = bytes;
        bytes_to_id_[bytes] = id;

        if (len == 1) {
            uint8_t byte_val = static_cast<uint8_t>(bytes[0]);
            byte_to_token_id[byte_val] = id;
        }
    }

    for (uint32_t i = 0; i < m_size; ++i) {
        TokenPair pair{};
        uint32_t rank = 0;
        TokenId result_id = 0;

        in.read(reinterpret_cast<char*>(&pair.first), sizeof(pair.first));
        in.read(reinterpret_cast<char*>(&pair.second), sizeof(pair.second));
        in.read(reinterpret_cast<char*>(&rank), sizeof(rank));
        in.read(reinterpret_cast<char*>(&result_id), sizeof(result_id));

        merge_ranks_[pair] = MergeInfo{rank, result_id};
    }
}

} // namespace tokenizer