cmake_minimum_required(VERSION 3.20)
project(BPE_Tokenizer LANGUAGES CXX)

add_library(tokenizer src/encoder.cpp)

target_include_directories(tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(tokenizer PUBLIC cxx_std_20)