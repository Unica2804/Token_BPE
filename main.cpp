#include "tokenizer/types.hpp"
#include "tokenizer/encoder.hpp"
#include "tokenizer/trainer.hpp"
#include "tokenizer/parallel.hpp"
#include <iostream>

int main() {
    using namespace tokenizer;
    BpeEncoder encoder;

    std::vector<WordItem> corpus = {
        {{'h', 'u', 'g'}, 10},
        {{'p', 'u', 'g'}, 5},
        {{'h', 'u', 'g', 's'}, 12},
        {{'b', 'u', 'g', 's'}, 4}
    };

    BpeTrainer::train(corpus,262,encoder);

    std::vector<std::string_view> text_chunks = {
        "hugs",
        "pugs",
        "hug"
    };
    auto tokenized_chunks = parallel_tokenize(encoder, text_chunks);
    for (size_t i = 0; i < text_chunks.size(); ++i) {
        std::cout << "Text: \"" << text_chunks[i] << "\" -> Tokens: ";
        for (TokenId id : tokenized_chunks[i]) {
            std::cout << id << " ";
        }
        std::cout << "| Decoded: \"" << encoder.decode(tokenized_chunks[i]) << "\"\n";
    }

    return 0;
}