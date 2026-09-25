#include "tokenizer/trainer.hpp"
#include "tokenizer/types.hpp"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace tokenizer {

void BpeTrainer::train(std::vector<WordItem> &corpus_words, size_t target_vocab_size, BpeEncoder &out_encoder) {
    TokenId next_token_id = 256;
    uint32_t current_rank = 0;

    std::unordered_map<TokenPair, std::unordered_set<size_t>, TokenPairHash> inverted_index;
    std::unordered_map<TokenPair, int64_t, TokenPairHash> pair_frequencies;

    // Build the inverted index and pair frequencies
    for (size_t w_idx = 0; w_idx < corpus_words.size(); ++w_idx) {
        const auto& tokens = corpus_words[w_idx].tokens;
        if (tokens.size() < 2) continue;
        
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            TokenPair p{tokens[i], tokens[i + 1]};
            pair_frequencies[p] += corpus_words[w_idx].frequency;
            inverted_index[p].insert(w_idx);
        }
    }

    while (next_token_id < target_vocab_size) {
        if (pair_frequencies.empty()) break;

        auto best_it = std::max_element(
            pair_frequencies.begin(), pair_frequencies.end(),
            [](const auto& a, const auto& b) {return a.second < b.second;}
        );

        if (best_it -> second <= 0) break;

        TokenPair best_pair = best_it->first;
        out_encoder.add_merge(best_pair.first, best_pair.second, current_rank++, next_token_id);
        auto affected_words = std::move(inverted_index[best_pair]);
        pair_frequencies.erase(best_it);
        inverted_index.erase(best_pair);

        // Update the inverted index
        for (size_t w_idx : inverted_index[best_pair]) {
            auto& words = corpus_words[w_idx];
            std::vector<TokenId> new_tokens;
            new_tokens.reserve(words.tokens.size());
            for (size_t i = 0; i < words.tokens.size() - 1; ++i) {
            
            }
        }

        ++next_token_id;
    }
}

}