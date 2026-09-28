#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/filesystem.h>

#include "tokenizer/encoder.hpp"
#include "tokenizer/parallel.hpp"

namespace nb = nanobind;
using namespace tokenizer;

NB_MODULE(fast_bpe, m) {
    nb::class_<BpeEncoder>(m, "BpeEncoder")
        .def(nb::init<>())
        .def("load", &BpeEncoder::load, nb::arg("filepath"))
        .def("save", &BpeEncoder::save, nb::arg("filepath"))
        .def("vocab_size", &BpeEncoder::vocab_size)
        // Single encode
        .def("encode", [](const BpeEncoder& self, std::string_view text) {
            return self.encode_chunk(text);
        })
        // Parallel batch encode (releases the GIL)
        .def("encode_batch", [](const BpeEncoder& self, const std::vector<std::string_view>& chunks) {
            nb::gil_scoped_release release;
            return parallel_tokenize(self, chunks);
        })
        .def("decode", &BpeEncoder::decode, nb::arg("tokens"));
}