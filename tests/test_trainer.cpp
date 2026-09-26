#include <catch2/catch_test_macros.hpp>
#include "tokenizer/trainer.hpp"
#include "tokenizer/encoder.hpp"

using namespace tokenizer;

TEST_CASE("Trainer handles boundary conditions on corpus", "[trainer]") {
    BpeEncoder encoder;

    SECTION("Empty corpus produces no merges") {
        std::vector<WordItem> empty_corpus;
        BpeTrainer::train(empty_corpus, 300, encoder);

        // Vocab must remain strictly the 256 base bytes
        REQUIRE(encoder.vocab_size() == 256);
    }

    SECTION("Words with fewer than two tokens are safely ignored") {
        std::vector<WordItem> single_char_corpus = {
            {{'a'}, 100},
            {{'b'}, 50}
        };

        // Attempt to train up to 260
        BpeTrainer::train(single_char_corpus, 260, encoder);
        
        // No merges possible from 1-length words
        REQUIRE(encoder.vocab_size() == 256);
    }

    SECTION("Target vocab size <= base vocab size terminates immediately") {
        std::vector<WordItem> corpus = {
            {{'a', 'b'}, 10}
        };

        // Base vocab is 256. Requesting 256 or less should perform zero merges.
        BpeTrainer::train(corpus, 256, encoder);
        REQUIRE(encoder.vocab_size() == 256);
        
        // Input should remain two separate tokens
        auto tokens = encoder.encode_chunk("ab");
        REQUIRE(tokens.size() == 2);
    }
}

TEST_CASE("Trainer stops cleanly when no frequent pairs remain", "[trainer]") {
    BpeEncoder encoder;

    std::vector<WordItem> corpus = {
        {{'c', 'a', 't'}, 5} // Only two pairs available: ('c','a') and ('a','t')
    };

    // Ask for 300 tokens, but corpus can at most yield 2 merges:
    // Merge 1: ('c', 'a') or ('a', 't')
    // Merge 2: the remaining combination into "cat"
    BpeTrainer::train(corpus, 300, encoder);

    // Trainer must terminate early without hanging or crashing
    REQUIRE(encoder.vocab_size() == 258);
    
    auto tokens = encoder.encode_chunk("cat");
    REQUIRE(tokens.size() == 1);
    REQUIRE(tokens[0] == 257); // The full word token
    REQUIRE(encoder.decode(tokens) == "cat");
}

TEST_CASE("Trainer maintains accurate pair counts across word frequency weights", "[trainer]") {
    BpeEncoder encoder;

    // 't' + 'o' appears:
    // - In word 1 ("to"): count 10
    // - In word 2 ("top"): count 20
    // Total 't'+'o' frequency = 30
    // 'o' + 'p' frequency = 20
    std::vector<WordItem> corpus = {
        {{'t', 'o'}, 10},
        {{'t', 'o', 'p'}, 20}
    };

    // Train exactly 1 merge (vocab 256 -> 257)
    BpeTrainer::train(corpus, 257, encoder);

    // Pair ('t', 'o') has higher frequency (30 > 20) and must be chosen as token 256
    auto tokens_to = encoder.encode_chunk("to");
    REQUIRE(tokens_to.size() == 1);
    REQUIRE(tokens_to[0] == 256);

    auto tokens_top = encoder.encode_chunk("top");
    REQUIRE(tokens_top.size() == 2);
    REQUIRE(tokens_top[0] == 256);
    REQUIRE(tokens_top[1] == static_cast<TokenId>('p'));
}