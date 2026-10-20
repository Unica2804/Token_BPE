#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/tuple.h>

#include "tokenizer/encoder.hpp"
#include "tokenizer/parallel.hpp"
#include <span>

namespace nb = nanobind;
using namespace tokenizer;

template <typename T>
nb::ndarray<nb::numpy,T,nb::ndim<1>> vector_to_ndarray(std::vector<T>&& vec) {
    auto* heap_vec = new std::vector<T>(std::move(vec));
    nb::capsule owner(heap_vec, [](void* p) noexcept {
        delete static_cast<std::vector<T>*>(p);
    });
    size_t shape[1] = { heap_vec->size() };
    return nb::ndarray<nb::numpy,T,nb::ndim<1>>(
        heap_vec->data(),
        1,
        shape,
        owner
    );
}

NB_MODULE(fast_bpe, m) {
    nb::class_<BpeEncoder>(m, "BpeEncoder")
        .def(nb::init<>())
        .def("load", &BpeEncoder::load, nb::arg("filepath"))
        .def("save", &BpeEncoder::save, nb::arg("filepath"))
        .def("vocab_size", &BpeEncoder::vocab_size)
        .def("merge_count", &BpeEncoder::merge_count)
        // Single encode
        .def("encode", [](const BpeEncoder& self, std::string_view text) {
            return self.encode_chunk(text);
        })
        //zero copy single encode
        .def("encode_flat", [](const BpeEncoder& self, std::string_view text) {
            std::vector<TokenId> tokens;
            {
                nb::gil_scoped_release release;
                tokens = self.encode_chunk(text);
            }
            return vector_to_ndarray(std::move(tokens));
        })
        // Parallel batch encode (releases the GIL)
        .def("encode_batch", [](const BpeEncoder& self, const std::vector<std::string_view>& chunks) {
            nb::gil_scoped_release release;
            return parallel_tokenize(self, chunks);
        })
        .def("encode_batch_flat", [](
            const BpeEncoder& self,
            const std::vector<std::string_view>& chunks,
            unsigned int threads
        ){
            FlatBatch flat;
            {
                nb::gil_scoped_release release;
                flat = parallel_tokenize_flat(self, chunks, threads);
            }
            auto tokens_arr = vector_to_ndarray(std::move(flat.tokens));
            auto offsets_arr = vector_to_ndarray(std::move(flat.offsets));
            return nb::make_tuple(tokens_arr, offsets_arr);
        }, nb::arg("chunks"), nb::arg("threads") = 0)
        
        .def("decode", [](
            const BpeEncoder& self,
            nb::ndarray<const TokenId, nb::ndim<1>, nb::c_contig> tokens
        ) {
            nb::gil_scoped_release release;
            return self.decode(std::span<const TokenId>(tokens.data(), tokens.size()));
        }, nb::arg("tokens"))

        .def("decode", [](
            const BpeEncoder& self,
            const std::vector<TokenId>& tokens
        ) {
            nb::gil_scoped_release release;
            return self.decode(std::span<const TokenId>(tokens.data(), tokens.size()));
        }, nb::arg("tokens"));

}