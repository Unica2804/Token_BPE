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