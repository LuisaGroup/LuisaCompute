#include "ut/ut.hpp"
#include "cuda_tile_codegen.h"
#include "cuda_tile_cub_scan.h"
#include <luisa/tile/algorithms.h>
#include <luisa/tile/verifier.h>
#include <luisa/core/stl/format.h>
#include <array>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::tile;
using namespace luisa::compute::cuda::native_tile;
using namespace boost::ut;

namespace {
enum class Change { NONE,
                    EPILOGUE,
                    EXTRA_USE,
                    CUSTOM_FILL };

template<typename T>
[[nodiscard]] tile::Kernel capture(int64_t rows, int64_t columns, int64_t width,
                                   int64_t block_rows = 1, Change change = Change::NONE,
                                   ReductionPolicy policy = reduction::unordered_tree) {
    return tile_kernel("unnamed_prefix_realization", [=](TensorView<const uint32_t, 1> unused0,
                                                         TensorView<const T, 2> input,
                                                         TensorView<const float, 1> unused2,
                                                         TensorView<T, 2> output,
                                                         TensorView<const int64_t, 1> unused4) {
               static_cast<void>(unused0);
               static_cast<void>(unused2);
               static_cast<void>(unused4);
               auto r = axis("independent", block_rows), c = axis("contribution", width);
               for (auto &program : parallel(shape((rows - 1) / block_rows + 1))) {
                   auto row = program.index() * block_rows;
                   auto memory = input.tile(coord(row, 0), shape(r, c));
                   auto x = cast<float>(change == Change::CUSTOM_FILL ? memory.load(static_cast<T>(1.0f)) : memory.load());
                   auto y = inclusive_sum(x, c, policy);
                   if (change == Change::EPILOGUE) { y = y + 1.0f; }
                   if (change == Change::EXTRA_USE) { y = y + x; }
                   output(coord(row, 0), shape(r, c)).store(cast<T>(y));
               }
           })
        .capture(tensor_shape(1), tensor_shape(rows, columns), tensor_shape(1), tensor_shape(rows, columns), tensor_shape(1));
}

[[nodiscard]] size_t count(luisa::string_view text, luisa::string_view needle) noexcept {
    auto result = size_t{0u};
    for (auto p = text.find(needle); p != luisa::string_view::npos; p = text.find(needle, p + needle.size())) { result++; }
    return result;
}

void source_and_abi() {
    auto check = []<typename T>() {
        for (auto rows : {int64_t{1}, int64_t{17}, int64_t{129}}) {
            for (auto threads : {128u, 256u, 512u, 1024u}) {
                for (auto chunks : {1u, 2u, 4u}) {
                    auto width = static_cast<int64_t>(8u * threads * chunks);
                    auto kernel = capture<T>(rows, width, width);
                    expect(kernel.valid());
                    if (!kernel.valid()) { continue; }
                    auto original = generate(kernel.function());
                    expect(original.ok()) << original.error;
                    if (!original.ok()) { continue; }
                    auto source_before = original.source;
                    auto candidate = generate_cub_scan(kernel.function(), original, threads);
                    expect(candidate.ok()) << candidate.error;
                    if (!candidate.ok()) { continue; }
                    expect(original.source == source_before);
                    expect(original.entry == "luisa_tile_main");
                    expect(original.block == std::array<uint32_t, 3u>{1u, 1u, 1u});
                    expect(candidate.entry == "luisa_tile_cub_scan");
                    expect(candidate.grid == std::array<uint32_t, 3u>{static_cast<uint32_t>(rows), 1u, 1u});
                    expect(candidate.block == std::array<uint32_t, 3u>{threads, 1u, 1u});
                    expect(candidate.threads == threads && candidate.chunk_extent == threads * 8u);
                    expect(candidate.guard.input_slot == 1u && candidate.guard.output_slot == 3u);
                    expect(candidate.guard.input_bytes == static_cast<uint64_t>(rows * width * sizeof(T)));
                    expect(candidate.guard.output_bytes == candidate.guard.input_bytes);
                    expect(candidate.alignment_mask == 0xau);
                    auto narrow = scalar_type_v<T> == ScalarType::FLOAT16 ? "__half" : "__nv_bfloat16";
                    auto codec = scalar_type_v<T> == ScalarType::FLOAT16 ? "Fp16Storage" : "Bf16Storage";
                    expect(candidate.source.find(luisa::format(
                               "void luisa_tile_cub_scan(unsigned *buffer0, {} *buffer1, float *buffer2, {} *buffer3, long long *buffer4)", narrow, narrow)) != string::npos);
                    expect(candidate.source.find(luisa::format("__launch_bounds__({})", threads)) != string::npos);
                    expect(candidate.source.find(luisa::format("RowScan<luisa_tile_cub_detail::{}, {}, {}>", codec, threads, width)) != string::npos);
                    expect(candidate.source.find("reinterpret_cast<const unsigned short *>(buffer1), reinterpret_cast<unsigned short *>(buffer3)") != string::npos);
                    expect(candidate.source.find("__tile_global__") == string::npos);
                    expect(candidate.source.find("restrict") == string::npos);
                    expect(candidate.source.find("ct::") == string::npos);
                    // The communication/rounding contract, not a mirrored
                    // implementation of either scan or source generation.
                    expect(count(candidate.source, "Scan{storage.scan}.InclusiveSum(values, values, aggregate);") == 1u);
                    expect(count(candidate.source, "__syncthreads();") == 1u);
                    expect(candidate.source.find("if (base != 0) { __syncthreads(); }") <
                           candidate.source.find("Scan{storage.scan}.InclusiveSum"));
                    expect(candidate.source.find("carry = base == 0 ? aggregate : __fadd_rn(carry, aggregate);") != string::npos);
                    expect(candidate.source.find("Codec::store(__fadd_rn(0.0f, cumulative))") != string::npos);
                    expect(candidate.source.find("static constexpr int Items = 8;") != string::npos);
                    expect(candidate.source.find("reinterpret_cast<const uint4 *>") != string::npos);
                    expect(candidate.source.find("reinterpret_cast<uint4 *>") != string::npos);
                }
            }
        }
    };
    check.template operator()<half>();
    check.template operator()<bfloat16>();
}

void eligibility_and_metadata() {
    auto reject = [](const tile::Kernel &kernel, uint32_t threads) {
        expect(kernel.valid());
        if (!kernel.valid()) { return; }
        auto original = generate(kernel.function());
        expect(original.ok()) << original.error;
        if (!original.ok()) { return; }
        auto before = original.source;
        auto candidate = generate_cub_scan(kernel.function(), original, threads);
        expect(!candidate.ok() && candidate.source.empty() && !candidate.error.empty());
        expect(original.source == before);
    };
    reject(capture<float>(17, 8192, 8192), 256u);
    reject(capture<half>(17, 8192, 8192, 2), 256u);
    reject(capture<half>(17, 8191, 8192), 256u);
    reject(capture<half>(17, 512, 512), 128u);
    for (auto change : {Change::EPILOGUE, Change::EXTRA_USE, Change::CUSTOM_FILL}) {
        reject(capture<half>(17, 8192, 8192, 1, change), 256u);
    }
    for (auto policy : {reduction::ordered_tree, reduction::fold_left, reduction::fold_right}) {
        reject(capture<half>(17, 8192, 8192, 1, Change::NONE, policy), 256u);
    }
    auto kernel = capture<half>(17, 8192, 8192);
    auto original = generate(kernel.function());
    expect(original.ok()) << original.error;
    if (!original.ok()) { return; }
    for (auto threads : {0u, 64u, 129u, 2048u, std::numeric_limits<uint32_t>::max()}) {
        expect(!generate_cub_scan(kernel.function(), original, threads).ok());
    }
    for (auto change = 0u; change < 12u; change++) {
        auto bad = original;
        switch (change) {
            case 0u: bad.arguments[1u].minimum_size_bytes--; break;
            case 1u: bad.arguments[3u].minimum_size_bytes++; break;
            case 2u: bad.arguments[1u].written = true; break;
            case 3u: bad.arguments[0u].read = true; break;
            case 4u: bad.arguments[2u].element = ScalarType::INT32; break;
            case 5u: bad.arguments.pop_back(); break;
            case 6u: bad.grid[0u]++; break;
            case 7u: bad.block[0u] = 256u; break;
            case 8u: bad.aligned16_buffer_mask = 2u; break;
            case 9u: bad.scan_chunk_extent = 1024u; break;
            case 10u: bad.streaming_scan_chunk_extent = 2048u; break;
            case 11u: bad.source.clear(); break;
        }
        auto candidate = generate_cub_scan(kernel.function(), bad, 256u);
        expect(!candidate.ok() && candidate.source.empty()) << change;
    }
    // Static root rows exceeding the physical grid limit cannot be truncated.
    auto huge = capture<half>(int64_t{1} << 32u, 2048, 2048);
    expect(huge.valid());
    auto proof = analyze_closed_prefix(huge.function());
    expect(proof.ok()) << proof.error;
    expect(!generate_cub_scan(huge.function(), original, 256u).ok());
}

void invocation_ranges() {
    auto kernel = capture<bfloat16>(17, 2048, 2048);
    auto original = generate(kernel.function());
    auto candidate = generate_cub_scan(kernel.function(), original, 256u);
    expect(candidate.ok()) << candidate.error;
    if (!candidate.ok()) { return; }
    std::array<uint64_t, 5u> pointers{1u, 0x1000u, 3u, 0x1000u + candidate.guard.input_bytes, 5u};
    auto admits = [&] { return streaming_scan_disjoint(candidate.guard, span<const uint64_t>{pointers}); };
    expect(admits());
    // Alignment applies only to the two actual participating arguments.
    auto aligned = [&] {
        uint64_t combined{};
        for (auto i = 0u; i < pointers.size(); i++) {
            if ((candidate.alignment_mask & (1u << i)) != 0u) { combined |= pointers[i]; }
        }
        return (combined & 15u) == 0u;
    };
    expect(aligned());
    pointers[3u] += 2u;
    expect(admits() && !aligned());
    pointers[3u] = pointers[1u];
    expect(!admits());
    pointers[3u] = pointers[1u] + candidate.guard.input_bytes - 16u;
    expect(!admits());
    pointers[1u] = std::numeric_limits<uint64_t>::max() - 15u;
    expect(!admits());
}
}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_cuda_cub_scan_source_and_abi"_test = source_and_abi;
    "tile_cuda_cub_scan_eligibility_and_metadata"_test = eligibility_and_metadata;
    "tile_cuda_cub_scan_invocation_ranges"_test = invocation_ranges;
}
