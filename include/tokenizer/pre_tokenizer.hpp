#pragma once

#include "swar.hpp"
#include <string_view>
#include <cctype>

namespace tokenizer {

    struct LexerCursor {
        const char* ptr{nullptr};
        const char* end{nullptr};
        [[nodiscard]] constexpr bool has_more() const noexcept {
            return ptr < end;
        }
        [[nodiscard]] constexpr size_t remaining() const noexcept {
            return static_cast<size_t>(end - ptr);
        }
    };

    inline std::string_view next_token(LexerCursor& cursor) {
        const char* start = cursor.ptr;
        const char* p = start;
        const char* end = cursor.end;
        const size_t rem = end - p;

        char c = *p;

        if (c == ' ' && rem > 1) {
            char next_c = p[1];
            if (std::isalpha(static_cast<unsigned char>(next_c)) ||
                std::isdigit(static_cast<unsigned char>(next_c)) ||
                (!std::isspace(static_cast<unsigned char>(next_c)) && next_c != '\'')) {
                p++;
                c = *p;
            }
        }

        if (std::isalpha(static_cast<unsigned char>(c))) {
            p++;
            while (p+8 <= end) {
                uint64_t v = swar::load64(p);
                if (!swar::all_ascii_letters(v)) {
                    break;
                }
                p += 8;
            }
            while (p < end && std::isalpha(static_cast<unsigned char>(*p))) {
                p++;
            }
            cursor.ptr = p;
            return std::string_view(start, p - start);
        }

        
        p = start;
        c= *p;

        if(std::isspace(static_cast<unsigned char>(c))) {
            p++;
            if (c == ' ') {
                while (p+8 <= end) {
                    uint64_t v = swar::load64(p);
                    if (!swar::all_bytes_equal(v, ' ')) {
                        break;
                    }
                    p += 8;
                }
            }
            while (p < end && std::isspace(static_cast<unsigned char>(*p))) {
                p++;
            }
            if((p - start) > 1 && p < end && *(p-1) == ' ' && *p != '\'') {
                p--;
            }
            
            cursor.ptr = p;
            return std::string_view(start, p - start);
        }

        if (std::isdigit(static_cast<unsigned char>(c))) {
            p++;
            while (p+8 <= end) {
                uint64_t v = swar::load64(p);
                if (!swar::all_ascii_digits(v)) {
                    break;
                }
                p += 8;
            }
            while (p < end && std::isdigit(static_cast<unsigned char>(*p))) {
                p++;
            }
            cursor.ptr = p;
            return std::string_view(start, p - start);
        }

        if (c=='\'') {
            if(rem >=2) {
                char n1 = p[1];
                if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                    cursor.ptr = start+2;
                    return std::string_view(start, 2);
                }
                if (rem >= 3) {
                    char n2 = p[2];
                    if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                        cursor.ptr = start+3;
                        return std::string_view(start, 3);
                    }
                }
            }
        }

        while (p<end && !std::isalnum(static_cast<unsigned char>(*p)) && 
    !std::isspace(static_cast<unsigned char>(*p)) && *p != '\'') {
            p++;
        }
        if (p == start) {
            p++;
        }
        cursor.ptr = p;
        return std::string_view(start, p - start);
    }
}