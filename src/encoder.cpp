#include "tokenizer/encoder.hpp"
#include <limits>
#include <iostream>

namespace tokenizer {

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

} // namespace tokenizer