#pragma once

#include "types.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

namespace tokenizer {

class TokenCache {
    public:
        static constexpr size_t MAX_CACHE_ENTRIES = 8192;
        static constexpr size_t MAX_CACHABLE_LENGTH = 32;

        TokenCache() {
            table_.reserve(MAX_CACHE_ENTRIES);
        }

        struct TransparentStringHash {
            using is_transparent = void;
            size_t operator()(std::string_view sv) const {
                return std::hash<std::string_view>{}(sv);
            }
        };

        struct TransparentStringEqual {
            using is_transparent = void;
            bool operator()(std::string_view lhs, std::string_view rhs) const noexcept {
                return lhs == rhs;
            }
        };

        using Table = std::unordered_map<
            std::string,
            std::vector<TokenId>,
            TransparentStringHash,
            TransparentStringEqual
        >;

        [[nodiscard]] const std::vector<TokenId>* get(std::string_view word) const noexcept {
            auto it = table_.find(word);
            if (it != table_.end()) {
                return &it->second;
            }
            return nullptr;
        }

        void put(std::string_view word, std::vector<TokenId> tokens) {
            if (word.size() > MAX_CACHABLE_LENGTH) {
                return;
            }
            if (table_.size() >= MAX_CACHE_ENTRIES) {
                table_.clear();
            }
            table_.emplace(std::string(word), std::move(tokens));
        }

        void clear() {
            table_.clear();
        }

        [[nodiscard]] size_t size() const noexcept {
            return table_.size();
        }
    private:
        Table table_;
};

}