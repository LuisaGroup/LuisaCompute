#include "ut/ut.hpp"

#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>

#include "spirv_codegen/buffer_layout.h"
#include "spirv_codegen/memory_access_plan.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int main() {
    "spirv_typed_buffer_layout_accepts_standard_host_layouts"_test = [] {
        auto *compatible_struct = Type::structure(
            16u, {Type::of<double2>(), Type::of<float4>()});
        auto *wide_vector_array = Type::array(
            Type::of<double4>(), 3u);
        auto *matrix_array = Type::array(
            Type::of<float2x2>(), 2u);
        auto *nested_matrix_struct = Type::structure(
            16u, {Type::of<float4>(), matrix_array,
                  Type::of<uint32_t>()});

        expect(lc::spirv::plan_spirv_typed_buffer_layout(
                   Type::of<double4>())
                   .compatible());
        expect(lc::spirv::plan_spirv_typed_buffer_layout(
                   compatible_struct)
                   .compatible());
        expect(lc::spirv::plan_spirv_typed_buffer_layout(
                   wide_vector_array)
                   .compatible());
        auto nested_matrix_layout =
            lc::spirv::plan_spirv_typed_buffer_layout(
                nested_matrix_struct);
        expect(nested_matrix_layout.compatible());
        expect(eq(nested_matrix_layout.base_alignment, 16u));
    };

    "spirv_typed_buffer_layout_rejects_host_vulkan_alignment_mismatch"_test = [] {
        auto *misaligned_member = Type::structure(
            16u, {Type::of<float4>(), Type::of<double4>()});
        auto member_layout =
            lc::spirv::plan_spirv_typed_buffer_layout(
                misaligned_member);
        expect(!member_layout.compatible());
        expect(member_layout.status ==
               lc::spirv::SpirvTypedBufferLayoutStatus::
                   MISALIGNED_STRUCT_MEMBER);
        expect(eq(member_layout.byte_offset, 16u));
        expect(member_layout.offending_type == Type::of<double4>());

        auto *invalid_struct_stride = Type::structure(
            16u, {Type::of<double4>(), Type::of<float>()});
        auto stride_layout =
            lc::spirv::plan_spirv_typed_buffer_layout(
                invalid_struct_stride);
        expect(!stride_layout.compatible());
        expect(stride_layout.status ==
               lc::spirv::SpirvTypedBufferLayoutStatus::
                   INVALID_STRUCT_STRIDE);

        auto bool_layout =
            lc::spirv::plan_spirv_typed_buffer_layout(
                Type::of<bool4>());
        expect(!bool_layout.compatible());
        expect(bool_layout.status ==
               lc::spirv::SpirvTypedBufferLayoutStatus::LOGICAL_BOOL);
    };

    // ---- exact word-access alignment proof (4B) --------------------------
    "spirv_aligned_word_access_only_uses_the_exact_proof"_test = [] {
        using lc::spirv::plan_spirv_aligned_word_access;
        constexpr auto none = spv::MemoryAccessMask::MaskNone;
        constexpr auto volatile_bits = spv::MemoryAccessMask::Volatile;

        // A four- or eight-byte proof annotates the word operand with 4.
        for (auto alignment : {size_t{4u}, size_t{8u}}) {
            auto plan = plan_spirv_aligned_word_access(none, alignment);
            expect(plan.aligned());
            expect(eq(plan.literal, 4u));
            expect(static_cast<bool>(plan.mask &
                                     spv::MemoryAccessMask::Aligned));
        }
        // One- and two-byte proofs are never rounded up, and neither is a
        // proof that is not a power of two.
        for (auto alignment : {size_t{1u}, size_t{2u}, size_t{3u}, size_t{6u}}) {
            auto plan = plan_spirv_aligned_word_access(none, alignment);
            expect(!plan.aligned()) << "alignment " << alignment;
            expect(eq(plan.literal, 0u));
            expect(!static_cast<bool>(plan.mask &
                                      spv::MemoryAccessMask::Aligned));
            expect(eq(static_cast<uint32_t>(plan.mask),
                      static_cast<uint32_t>(none)));
        }
        // Every existing bit is preserved: Aligned is additive only.
        auto volatile_plan =
            plan_spirv_aligned_word_access(volatile_bits, 4u);
        expect(volatile_plan.aligned());
        expect(static_cast<bool>(volatile_plan.mask &
                                 spv::MemoryAccessMask::Volatile));
        expect(static_cast<bool>(volatile_plan.mask &
                                 spv::MemoryAccessMask::Aligned));
        auto unproven_volatile =
            plan_spirv_aligned_word_access(volatile_bits, 1u);
        expect(!unproven_volatile.aligned());
        expect(eq(static_cast<uint32_t>(unproven_volatile.mask),
                  static_cast<uint32_t>(volatile_bits)))
            << "an unproven access must keep its exact original mask";
        // A wide scalar never gets a literal smaller than the access.
        expect(!plan_spirv_aligned_word_access(none, 8u, 8u).aligned());
    };
}
