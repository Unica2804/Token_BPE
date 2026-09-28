#include "tokenizer/trainer.hpp"
#include "tokenizer/types.hpp"
#include <unordered_map>
#include <algorithm>

namespace tokenizer {

void BpeTrainer::train(std::vector<WordItem> &corpus_words, size_t target_vocab_size, BpeEncoder &out_encoder) {
    TokenId next_token_id = 256;
    uint32_t current_rank = 0;

    std::unordered_map<TokenPair, int64_t, TokenPairHash> pair_frequencies;

    auto rebuild_pair_frequencies = [&]() {
        pair_frequencies.clear();
        for (const auto& word : corpus_words) {
            const auto& tokens = word.tokens;
            if (tokens.size() < 2) continue;

            for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                pair_frequencies[{tokens[i], tokens[i + 1]}] += word.frequency;
            }
        }
    };

    rebuild_pair_frequencies();

    while (next_token_id < target_vocab_size) {
        if (pair_frequencies.empty()) break;

        auto best_it = std::max_element(
            pair_frequencies.begin(), pair_frequencies.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );

        if (best_it->second <= 0) break;

        const TokenPair best_pair = best_it->first;
        out_encoder.add_merge(best_pair.first, best_pair.second, current_rank++, next_token_id);

        for (auto& word : corpus_words) {
            const auto& tokens = word.tokens;
            std::vector<TokenId> new_tokens;
            new_tokens.reserve(tokens.size());

            for (size_t i = 0; i < tokens.size();) {
                if (i + 1 < tokens.size() &&
                    tokens[i] == best_pair.first &&
                    tokens[i + 1] == best_pair.second) {
                    new_tokens.push_back(next_token_id);
                    i += 2;
                } else {
                    new_tokens.push_back(tokens[i]);
                    ++i;
                }
            }

            word.tokens = std::move(new_tokens);
        }

        ++next_token_id;
        rebuild_pair_frequencies();
    }
}

}