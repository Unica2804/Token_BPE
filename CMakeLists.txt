cmake_minimum_required(VERSION 3.20)
project(bpe_tokenizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
elseif (MSVC)
    add_compile_options(/W4 /O2 /permissive-)
endif()

# Core static library
add_library(bpe_core STATIC
    src/encoder.cpp
    src/trainer.cpp
)

target_include_directories(bpe_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Main demo executable
add_executable(tokenizer_cli main.cpp)
target_link_libraries(tokenizer_cli PRIVATE bpe_core)

# -------------------------------------------------------------
# Catch2 Test Suite Integration
# -------------------------------------------------------------
include(CTest)
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2 # Catch2 v3 stable release
)
FetchContent_MakeAvailable(Catch2)

add_executable(tokenizer_tests
    tests/test_encoder.cpp
    tests/test_trainer.cpp
)

target_link_libraries(tokenizer_tests PRIVATE 
    bpe_core 
    Catch2::Catch2WithMain
)

# Register with CTest
include(Catch)
catch_discover_tests(tokenizer_tests)