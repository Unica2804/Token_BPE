#include <catch2/catch_test_macros.hpp>
#include "tokenizer/encoder.hpp"

using namespace tokenizer;

TEST_CASE("Encoder handles empty and single-byte inputs", "[encoder]") {
    BpeEncoder encoder;

    SECTION("Empty input returns empty token vector") {
        auto tokens = encoder.encode_chunk("");
        REQUIRE(tokens.empty());
        REQUIRE(encoder.decode(tokens).empty());
    }

    SECTION("Single byte returns raw base token without merges") {
        auto tokens = encoder.encode_chunk("a");
        REQUIRE(tokens.size() == 1);
        REQUIRE(tokens[0] == static_cast<TokenId>('a'));
        REQUIRE(encoder.decode(tokens) == "a");
    }
}

TEST_CASE("Encoder correctly roundtrips arbitrary byte values", "[encoder]") {
    BpeEncoder encoder;

    SECTION("UTF-8 multibyte characters encode and decode identically") {
        // "こんにちは" (Hello in Japanese) consists of 3-byte UTF-8 sequences each
        std::string_view utf8_text = "こんにちは";
        auto tokens = encoder.encode_chunk(utf8_text);
        
        // Without merges, token count must match exact byte size
        REQUIRE(tokens.size() == utf8_text.size());
        REQUIRE(encoder.decode(tokens) == utf8_text);
    }

    SECTION("Binary null characters and high bytes roundtrip accurately") {
        std::string binary_data;
        binary_data.push_back('\0');
        binary_data.push_back(static_cast<char>(0xFF));
        binary_data.push_back(static_cast<char>(0x7F));

        auto tokens = encoder.encode_chunk(binary_data);
        REQUIRE(tokens.size() == 3);
        REQUIRE(tokens[0] == 0);
        REQUIRE(tokens[1] == 255);
        REQUIRE(tokens[2] == 127);
        REQUIRE(encoder.decode(tokens) == binary_data);
    }
}

TEST_CASE("Encoder respects merge priority rank over traversal order", "[encoder]") {
    BpeEncoder encoder;
    
    // String: "ABC"
    // Pair (B, C) learned with rank 0 -> new token 256
    // Pair (A, B) learned with rank 1 -> new token 257
    encoder.add_merge('B', 'C', 0, 256);
    encoder.add_merge('A', 'B', 1, 257);

    auto tokens = encoder.encode_chunk("ABC");

    // Even though (A, B) appears first in the string, (B, C) has a lower rank (0 < 1).
    // The result must be: 'A' followed by 256 ("BC"), NOT 257 ("AB") followed by 'C'.
    REQUIRE(tokens.size() == 2);
    REQUIRE(tokens[0] == static_cast<TokenId>('A'));
    REQUIRE(tokens[1] == 256);
    REQUIRE(encoder.decode(tokens) == "ABC");
}

TEST_CASE("Repeated identical tokens merge deterministically from left-to-right", "[encoder]") {
    BpeEncoder encoder;

    // Merge: 'a' + 'a' -> 256
    encoder.add_merge('a', 'a', 0, 256);

    SECTION("Odd repetition 'aaa' results in [256, 'a']") {
        auto tokens = encoder.encode_chunk("aaa");
        REQUIRE(tokens.size() == 2);
        REQUIRE(tokens[0] == 256);
        REQUIRE(tokens[1] == static_cast<TokenId>('a'));
        REQUIRE(encoder.decode(tokens) == "aaa");
    }

    SECTION("Even repetition 'aaaa' results in [256, 256]") {
        auto tokens = encoder.encode_chunk("aaaa");
        REQUIRE(tokens.size() == 2);
        REQUIRE(tokens[0] == 256);
        REQUIRE(tokens[1] == 256);
        REQUIRE(encoder.decode(tokens) == "aaaa");
    }
}

TEST_CASE("PretokenCache correctly memoizes and matches uncached output", "[cache]") {
    BpeEncoder encoder;
    encoder.add_merge('h', 'u', 0, 256);
    encoder.add_merge(256, 'g', 1, 257);

    std::string_view word = "hug";

    // 1st run: Cache miss -> computes and caches
    auto first_run = encoder.encode_chunk(word);
    
    // 2nd run: Cache hit -> fetched from memory
    auto second_run = encoder.encode_chunk(word);

    // Uncached direct evaluation
    auto direct_run = encoder.encode_chunk_uncached(word);

    REQUIRE(first_run == direct_run);
    REQUIRE(second_run == direct_run);
}

TEST_CASE("ThreadScratchpad reliably resets across consecutive chunk calls", "[scratchpad]") {
    BpeEncoder encoder;
    encoder.add_merge('a', 'b', 0, 256);
    encoder.add_merge('b', 'c', 1, 257);

    std::vector<TokenId> out1;
    encoder.encode_chunk_to_buffer("abc", out1);
    // 'a' + 'b' -> 256; then 'c' remains
    REQUIRE(out1.size() == 2);
    REQUIRE(out1[0] == 256);
    REQUIRE(out1[1] == static_cast<TokenId>('c'));

    // Call with a shorter chunk: verify previous larger scratchpad nodes do not leak
    std::vector<TokenId> out2;
    encoder.encode_chunk_to_buffer("a", out2);
    REQUIRE(out2.size() == 1);
    REQUIRE(out2[0] == static_cast<TokenId>('a'));

    // Call with a longer chunk
    std::vector<TokenId> out3;
    encoder.encode_chunk_to_buffer("ababc", out3);
    // "ab" (256) + "ab" (256) + "c"
    REQUIRE(out3.size() == 3);
    REQUIRE(out3[0] == 256);
    REQUIRE(out3[1] == 256);
    REQUIRE(out3[2] == static_cast<TokenId>('c'));
}

TEST_CASE("Encoder serialization roundtrips flawlessly", "[serialization]") {
    BpeEncoder original;
    original.add_merge('a', 'b', 0, 256);
    original.add_merge(256, 'c', 1, 257);

    const std::filesystem::path temp_file = "test_model.bpe";
    original.save(temp_file);

    BpeEncoder loaded;
    loaded.load(temp_file);
    std::filesystem::remove(temp_file);

    REQUIRE(loaded.vocab_size() == original.vocab_size());
    REQUIRE(loaded.merge_count() == original.merge_count());

    // Both should yield identical token sequences
    auto orig_tokens = original.encode_chunk("abc");
    auto loaded_tokens = loaded.encode_chunk("abc");

    REQUIRE(orig_tokens == loaded_tokens);
    REQUIRE(loaded.decode(loaded_tokens) == "abc");
}