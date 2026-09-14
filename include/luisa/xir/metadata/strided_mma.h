#pragma once

#include <cstdint>
#include <limits>

#include <luisa/core/stl/vector.h>
#include <luisa/xir/metadata.h>

namespace luisa::compute::xir {

enum struct StridedMmaVectorization : uint8_t {
    OUTPUT,
    CONTRACTION,
};

[[nodiscard]] constexpr luisa::string_view to_string(StridedMmaVectorization value) noexcept {
    switch (value) {
        case StridedMmaVectorization::OUTPUT: return "output";
        case StridedMmaVectorization::CONTRACTION: return "contraction";
    }
    return "unknown";
}

struct StridedMmaDescriptor {
    luisa::vector<uint64_t> output_extents;
    luisa::vector<uint64_t> lhs_output_strides;
    luisa::vector<uint64_t> rhs_output_strides;
    uint64_t contraction_extent{0u};
    uint64_t lhs_contraction_stride{0u};
    uint64_t rhs_contraction_stride{0u};
    uint32_t vector_width{1u};
    bool allow_reassociation{false};
    StridedMmaVectorization vectorization{StridedMmaVectorization::OUTPUT};
};

// Structural validation only: targets additionally validate the four reference
// types/capacities, local storage, supported width, and vectorized stride mode.
// In particular, this does not establish that a backend supports the intrinsic.
[[nodiscard]] inline bool is_valid_strided_mma_descriptor(const StridedMmaDescriptor &d) noexcept {
    auto rank = d.output_extents.size();
    if (rank == 0u || d.lhs_output_strides.size() != rank ||
        d.rhs_output_strides.size() != rank || d.vector_width == 0u) {
        return false;
    }
    switch (d.vectorization) {
        case StridedMmaVectorization::OUTPUT: break;
        case StridedMmaVectorization::CONTRACTION:
            if (!d.allow_reassociation) { return false; }
            break;
        default: return false;
    }
    constexpr auto limit = std::numeric_limits<uint64_t>::max();
    auto volume = uint64_t{1u};
    auto lhs_offset = uint64_t{0u};
    auto rhs_offset = uint64_t{0u};
    auto accumulate_offset = [](uint64_t &offset, uint64_t index, uint64_t stride) noexcept {
        if (stride != 0u && index > (limit - offset) / stride) { return false; }
        offset += index * stride;
        return true;
    };
    for (auto i = size_t{0u}; i < rank; i++) {
        auto extent = d.output_extents[i];
        if (extent == 0u || volume > limit / extent) { return false; }
        volume *= extent;
        if (d.contraction_extent != 0u &&
            (!accumulate_offset(lhs_offset, extent - 1u, d.lhs_output_strides[i]) ||
             !accumulate_offset(rhs_offset, extent - 1u, d.rhs_output_strides[i]))) {
            return false;
        }
    }
    // A zero-length contraction reads no operand element and copies the seed.
    if (d.contraction_extent == 0u) { return true; }
    return accumulate_offset(lhs_offset, d.contraction_extent - 1u, d.lhs_contraction_stride) &&
           accumulate_offset(rhs_offset, d.contraction_extent - 1u, d.rhs_contraction_stride) &&
           lhs_offset != limit && rhs_offset != limit;
}

// Required native-intrinsic semantics on a compiler-owned ExternalFunction,
// not an ignorable optimization hint or a function-name convention. The void
// call takes four local fixed-array<float> references: (lhs, rhs, seed, output).
// Output/seed are row-major in output_extents. For each output coordinate o:
//   acc = seed[flatten(o)];
//   for k = 0 .. contraction_extent-1:
//     acc = acc + lhs[dot(o, lhs_output_strides) + k * lhs_contraction_stride]
//               * rhs[dot(o, rhs_output_strides) + k * rhs_contraction_stride];
//   output[flatten(o)] = acc;
// Without allow_reassociation, FP32 multiply then add and ascending k order
// are preserved (no FMA contraction). With it, reduction reassociation is
// permitted; multiplication and addition remain separate FP32 operations.
// Lhs/rhs/seed are read-only snapshots for the call; output must not alias any
// input reference. The three inputs may alias one another. K=0 copies the seed
// exactly.
// Backends must reject unsupported descriptors/calls, including an external
// whose semantic metadata was removed; they must not infer it from its name.
class LUISA_XIR_API StridedMmaMD final
    : public DerivedMetadata<StridedMmaMD, DerivedMetadataTag::STRIDED_MMA> {
public:
    StridedMmaDescriptor descriptor;

    StridedMmaMD() noexcept = default;
    explicit StridedMmaMD(StridedMmaDescriptor descriptor) noexcept;
    [[nodiscard]] ManagedPtr<Metadata> clone() const noexcept override;
};

}// namespace luisa::compute::xir
