#include "ut/ut.hpp"

#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>

#include "spirv_codegen/buffer_layout.h"

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
}
