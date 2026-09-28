#include "tokenizer/encoder.hpp"
#include <fstream>
#include "stdexcept"
#include <array>
#include <limits>
#include <iostream>

namespace tokenizer {

    namespace {
        constexpr std::array<char, 4> MAGIC = {'B', 'P', 'E', '1'};
        constexpr uint8_t FORMAT_VERSION = 1;
    }

    BpeEncoder::BpeEncoder() {
        for (uint32_t i = 0; i < 256; ++i) {
            std::string byte_str(1, static_cast<char>(i));
            id_to_bytes_[i] = byte_str;
            bytes_to_id_[byte_str] = i;
        }
    }

    void BpeEncoder::add_merge(TokenId left, TokenId right, uint32_t rank, TokenId new_token_id) {
        merge_ranks_[{left, right}] = rank;
        // Give this new token id a string representation
        std::string merged_bytes = id_to_bytes_[left] + id_to_bytes_[right];
        id_to_bytes_[new_token_id] = merged_bytes;
        bytes_to_id_[merged_bytes] = new_token_id;
    }

    std::vector<TokenId> BpeEncoder::encode_chunk(std::string_view chunk) const {
        if (chunk.empty()) {
            return {};
        }

        const size_t n = chunk.size();
        std::vector<ListNode> nodes(n);
        for (size_t i = 0; i < n; ++i) {
            nodes[i].token = static_cast<uint8_t>(chunk[i]);
            nodes[i].prev = static_cast<int32_t>(i) - 1;
            nodes[i].next = static_cast<int32_t>(i) + 1;
        }
        // Last node has no next
        nodes.back().next = -1;

        while(true) {
            uint32_t lowest_rank = std::numeric_limits<uint32_t>::max();
            int32_t best_node_index = -1;

            int32_t curr = 0;
            while (curr != -1 && nodes[curr].next != -1) {
                int32_t next = nodes[curr].next;
                TokenPair pair{nodes[curr].token, nodes[next].token};
                if(auto it = merge_ranks_.find(pair); it != merge_ranks_.end()) {
                    if (it->second < lowest_rank) {
                        lowest_rank = it->second;
                        best_node_index = curr;
                    }
                }
                curr = next;
            }
            // If no valid pair was found, we are done
            if (best_node_index == -1) break;

            //Apply in-place merge
            int32_t merge_left = best_node_index;
            int32_t merge_right = nodes[merge_left].next;
            int32_t right_neighbour = nodes[merge_right].next;

            TokenPair target_pair{nodes[merge_left].token, nodes[merge_right].token};

            // Reconstruct the merged token id
            std::string combined = id_to_bytes_.at(target_pair.first) + id_to_bytes_.at(target_pair.second);
            nodes[merge_left].token = bytes_to_id_.at(combined);

            //bypass the right node (O(1) unlink)
            nodes[merge_left].next = right_neighbour;
            if (right_neighbour != -1) {
                nodes[right_neighbour].prev = merge_left;
            }
        }
        // Collect the final tokens
        std::vector<TokenId> result;
        int32_t curr = 0;
        while (curr != -1) {
            result.push_back(nodes[curr].token);
            curr = nodes[curr].next;
        }
        return result;
    }
    std::string BpeEncoder::decode(const std::vector<TokenId>& tokens) const {
        std::string result;
        for (TokenId token : tokens) {
            // If token is not equals not found we add it
            if(auto it = id_to_bytes_.find(token); it != id_to_bytes_.end()) {
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

        // Header
        out.write(MAGIC.data(), MAGIC.size());
        out.write(reinterpret_cast<const char*>(&FORMAT_VERSION), sizeof(FORMAT_VERSION));

        uint32_t v_size = static_cast<uint32_t>(id_to_bytes_.size());
        uint32_t m_size = static_cast<uint32_t>(merge_ranks_.size());
        out.write(reinterpret_cast<const char*>(&v_size), sizeof(v_size));
        out.write(reinterpret_cast<const char*>(&m_size), sizeof(m_size));

        // vocab table
        for (const auto& [id, bytes] : id_to_bytes_) {
            out.write(reinterpret_cast<const char*>(&id),sizeof(id));
            uint32_t bytes_size = static_cast<uint32_t>(bytes.size());
            out.write(reinterpret_cast<const char*>(&bytes_size), sizeof(bytes_size));
            out.write(bytes.data(), bytes_size);
        }

        // merge table
        for (const auto& [pair, rank]: merge_ranks_) {
            out.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
            out.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
            out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));

            // lookup result id
            std::string merged_bytes = id_to_bytes_.at(pair.first) + id_to_bytes_.at(pair.second);
            TokenId result_id = bytes_to_id_.at(merged_bytes);
            out.write(reinterpret_cast<const char*>(&result_id), sizeof(result_id));
        }
    }

    // Load the encoder state from a binary file
    void BpeEncoder::load(const std::filesystem::path& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + filepath.string());
        }

        // Verify header
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

        // Reset current state
        id_to_bytes_.clear();
        bytes_to_id_.clear();
        merge_ranks_.clear();

        // read vocab table
        for (uint32_t i = 0; i < v_size; ++i) {
            TokenId id = 0;
            uint32_t len = 0;
            in.read(reinterpret_cast<char*>(&id), sizeof(id));
            in.read(reinterpret_cast<char*>(&len), sizeof(len));

            std::string bytes(len, '\0');
            in.read(bytes.data(), len);

            id_to_bytes_[id] = bytes;
            bytes_to_id_[bytes] = id;
        }

        // read merge table
        for (uint32_t i = 0; i < m_size; ++i) {
            TokenPair pair{};
            uint32_t rank = 0;
            TokenId result_id = 0;

            in.read(reinterpret_cast<char*>(&pair.first), sizeof(pair.first));
            in.read(reinterpret_cast<char*>(&pair.second), sizeof(pair.second));
            in.read(reinterpret_cast<char*>(&rank), sizeof(rank));
            in.read(reinterpret_cast<char*>(&result_id), sizeof(result_id));

            merge_ranks_[pair] = rank;
        }
    }

} // namespace tokenizer