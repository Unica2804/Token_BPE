#pragma once

#include "tokenizer/types.hpp"
#include "tokenizer/encoder.hpp"
#include <vector>

namespace tokenizer {

struct WordItem {
    std::vector<TokenId> tokens;
    uint32_t frequency{1};
};

class BpeTrainer {
public:
    static void train(
        std::vector<WordItem>& corpus_words,
        size_t target_vocab_size,
        BpeEncoder& out_encoder
    );
};

}// namespace tokenizer