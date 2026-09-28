cmake_minimum_required(VERSION 3.20)
project(bpe_tokenizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # -O3, native vectorization, and link-time optimization (LTO)
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
elseif (MSVC)
    add_compile_options(/W4 /O2 /permissive-)
endif()

# Core static library
add_library(bpe_core STATIC
    src/encoder.cpp
    src/trainer.cpp
)
set_target_properties(bpe_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(bpe_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Main demo executable
add_executable(tokenizer_cli main.cpp)
target_link_libraries(tokenizer_cli PRIVATE bpe_core)

# Benchmark executable
add_executable(tokenizer_bench bench/benchmark.cpp)
target_link_libraries(tokenizer_bench PRIVATE bpe_core)

# -------------------------------------------------------------
# Catch2 Test Suite Integration
# -------------------------------------------------------------
include(CTest)
include(FetchContent)

if (NOT Python_EXECUTABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    set(Python_EXECUTABLE
        "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python"
        CACHE FILEPATH "Python interpreter used for nanobind"
    )
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(fast_bpe utils/bindings.cpp)
target_link_libraries(fast_bpe PRIVATE bpe_core)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
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

include(Catch)
catch_discover_tests(tokenizer_tests)