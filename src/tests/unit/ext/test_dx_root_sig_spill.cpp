// Focused unit tests for DX >64-DWORD root-signature descriptor-table spill
// property-to-argument mapping (`lc::dx::build_root_sig_layout`).
//
// Regression coverage for the BMW27 `shade_surface` DX hang where
// `PreprocessCodegenProperties` emits the uniform argument buffer `_Global` as
// `StructuredBuffer` at (space=0, reg=0) followed by bindless descriptor heaps
// (`array_size == UINT_MAX`). Failing to keep `t0, space0` direct and failing
// to skip `_Global` + bindless heaps shifted every spilled kernel buffer's
// `StructureByteStride` and raw-buffer flag by 3 slots.

#include "ut/ut.hpp"

#include <cstdint>
#include <limits>
#include <luisa/core/stl/filesystem.h>
#include <luisa/dsl/sugar.h>
#include "hlsl_codegen.h"
#include "shader_compiler.h"
#include "Shader/RootSigLayout.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

struct TestUniformBlob {
    uint a;
    uint b;
    float4 c;
};

struct TestPackedStruct {
    uint4 a;
    uint4 b;
    uint tag;
};

struct TestAttributeMap8Bit {
    uint id;
    uint8_t element;
    uint8_t type;
    uint8_t flags;
    int8_t signed_tag;
    uint offset;
};

struct TestAliasedPacked8Bit {
    uint id;
    uint8_t a;
    uint8_t b;
    int8_t c;
    uint8_t d;
    bool3 mask;
    uint tail;
};

LUISA_STRUCT(TestUniformBlob, a, b, c) {};
LUISA_STRUCT(TestPackedStruct, a, b, tag) {};
LUISA_STRUCT(TestAttributeMap8Bit, id, element, type, flags, signed_tag, offset) {};
LUISA_STRUCT(TestAliasedPacked8Bit, id, a, b, c, d, mask, tail) {};

namespace {

using lc::dx::RootSigBinding;
using lc::dx::SavedArgument;
using lc::hlsl::Property;
using lc::hlsl::ShaderVariableType;

constexpr uint kUnbounded = std::numeric_limits<uint>::max();

[[nodiscard]] SavedArgument make_buffer_arg(Usage usage, uint stride, bool raw = false) noexcept {
    SavedArgument a{};
    a.tag = Type::Tag::BUFFER;
    a.var_usage = usage;
    a.is_raw_byte_buffer = raw;
    a.struct_size = raw ? 1u : stride;
    return a;
}

[[nodiscard]] SavedArgument make_uniform_arg(uint bytes = 4u) noexcept {
    SavedArgument a{};
    a.tag = Type::Tag::UINT32;
    a.var_usage = Usage::READ;
    a.is_raw_byte_buffer = false;
    a.struct_size = bytes;
    return a;
}

[[nodiscard]] SavedArgument make_bindless_arg() noexcept {
    SavedArgument a{};
    a.tag = Type::Tag::BINDLESS_ARRAY;
    a.var_usage = Usage::READ;
    a.struct_size = 0u;
    return a;
}

[[nodiscard]] SavedArgument make_accel_arg(Usage usage) noexcept {
    SavedArgument a{};
    a.tag = Type::Tag::ACCEL;
    a.var_usage = usage;
    a.struct_size = 0u;
    return a;
}

[[nodiscard]] SavedArgument make_texture_arg(Usage usage) noexcept {
    SavedArgument a{};
    a.tag = Type::Tag::TEXTURE;
    a.var_usage = usage;
    a.struct_size = 0u;
    return a;
}

}// namespace

int main(int argc, char **argv) {
    static_cast<void>(argc);
    "below_and_above_64_dword_threshold"_test = [] {
        vstd::vector<Property> small_props;
        small_props.emplace_back(Property{ShaderVariableType::ConstantValue, 4u, 0u, 1u});
        small_props.emplace_back(Property{ShaderVariableType::SamplerHeap, 1u, 0u, 16u});
        for (uint i = 0u; i < 29u; ++i) {
            small_props.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, i, 1u});
        }
        // 4 + 1 + 29*2 = 63 DWORDs <= 64
        expect(!lc::dx::needs_descriptor_table_spill(small_props, false));
        expect(!lc::dx::needs_descriptor_table_spill(small_props, true));

        small_props.emplace_back(Property{ShaderVariableType::RWStructuredBuffer, 0u, 0u, 1u});
        // 65 DWORDs > 64
        expect(lc::dx::needs_descriptor_table_spill(small_props, false));
    };

    "uniform_structured_buffer_and_bindless_prefix_mapping"_test = [] {
        // Mirrors BMW27 `shade_surface`:
        //   prop[0] = ConstantValue (dsp_c)
        //   prop[1] = SamplerHeap
        //   prop[2] = StructuredBuffer (t0, space0) -> uniform arg buffer _Global
        //   prop[3] = SRVBufferHeap (bindless, space1, arr=UINT_MAX)
        //   prop[4] = SRVTextureHeap (bindless, space2, arr=UINT_MAX)
        //   prop[5..] = kernel resource arguments
        vstd::vector<Property> props;
        props.emplace_back(Property{ShaderVariableType::ConstantValue, 4u, 0u, 1u});
        props.emplace_back(Property{ShaderVariableType::SamplerHeap, 1u, 0u, 16u});
        props.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, 0u, 1u});
        props.emplace_back(Property{ShaderVariableType::SRVBufferHeap, 1u, 0u, kUnbounded});
        props.emplace_back(Property{ShaderVariableType::SRVTextureHeap, 2u, 0u, kUnbounded});

        vstd::vector<SavedArgument> args;
        uint srv_reg = 1u;
        uint uav_reg = 0u;

        // arg[0]: RWByteAddressBuffer (frame_buffer) -> prop[5]
        props.emplace_back(Property{ShaderVariableType::RWStructuredBuffer, 0u, uav_reg++, 1u});
        args.emplace_back(make_buffer_arg(Usage::READ_WRITE, 1u, true));
        // arg[1]: RWStructuredBuffer<uint> (_resume_count) -> prop[6]
        props.emplace_back(Property{ShaderVariableType::RWStructuredBuffer, 0u, uav_reg++, 1u});
        args.emplace_back(make_buffer_arg(Usage::WRITE, 4u));
        // arg[2]: Accel (read-only: TopAccel + 64B InstBuffer) -> prop[7], prop[8]
        props.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, srv_reg++, 1u});
        props.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, srv_reg++, 1u});
        args.emplace_back(make_accel_arg(Usage::READ));
        // arg[3]: BindlessArray -> prop[9]
        props.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, srv_reg++, 1u});
        args.emplace_back(make_bindless_arg());
        // arg[4]: uniform scalar (produces 0 properties)
        args.emplace_back(make_uniform_arg(4u));
        // arg[5]: Buffer<112B struct> (READ) -> prop[10]
        props.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, srv_reg++, 1u});
        args.emplace_back(make_buffer_arg(Usage::READ, 112u));
        // arg[6]: unused buffer (Usage::NONE, still emits SRV) -> prop[11]
        props.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, srv_reg++, 1u});
        args.emplace_back(make_buffer_arg(Usage::NONE, 52u));
        // Fill enough buffers to exceed 64 DWORDs (30 more buffers -> 60+ DWORDs)
        for (uint k = 0u; k < 28u; ++k) {
            const uint stride = (k % 2u == 0u) ? 16u : 12u;
            props.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, srv_reg++, 1u});
            args.emplace_back(make_buffer_arg(Usage::READ, stride));
        }

        auto layout = lc::dx::build_root_sig_layout(props, args, false, false);
        expect(layout.spilled_tables);
        expect(layout.mapping_complete);
        expect(layout.kernel_arg_prop_offset == 5_u);

        // Prefix properties MUST stay Direct in exact root-param order 0..4
        for (size_t i = 0u; i < 5u; ++i) {
            expect(layout.bindings[i].kind == RootSigBinding::Kind::Direct);
            expect(layout.bindings[i].index == i);
        }
        // Uniform StructuredBuffer (prop[2]) MUST stay Direct (not spilled)
        expect(!lc::dx::is_spilled_srv(props[2]));
        expect(layout.bindings[2].kind == RootSigBinding::Kind::Direct);

        // prop[5]: frame_buffer (ByteBuffer) -> SpilledUAV[0], raw=true
        expect(layout.bindings[5].kind == RootSigBinding::Kind::SpilledUAV);
        expect(layout.bindings[5].index == 0_u);
        expect(layout.bindings[5].force_raw);
        expect(layout.bindings[5].element_stride == 1_u);

        // prop[6]: _resume_count (Buffer<uint>) -> SpilledUAV[1], stride=4, raw=false
        expect(layout.bindings[6].kind == RootSigBinding::Kind::SpilledUAV);
        expect(layout.bindings[6].index == 1_u);
        expect(!layout.bindings[6].force_raw);
        expect(layout.bindings[6].element_stride == 4_u);

        // prop[7..8]: Accel -> SpilledSRV[0] (TopAccel stride=0), SpilledSRV[1] (Inst stride=64)
        expect(layout.bindings[7].kind == RootSigBinding::Kind::SpilledSRV);
        expect(layout.bindings[7].index == 0_u);
        expect(layout.bindings[7].element_stride == 0_u);
        expect(layout.bindings[8].kind == RootSigBinding::Kind::SpilledSRV);
        expect(layout.bindings[8].index == 1_u);
        expect(layout.bindings[8].element_stride == 64_u);

        // prop[9]: BindlessArray -> SpilledSRV[2], stride=4
        expect(layout.bindings[9].kind == RootSigBinding::Kind::SpilledSRV);
        expect(layout.bindings[9].index == 2_u);
        expect(layout.bindings[9].element_stride == 4_u);

        // prop[10]: 112B struct buffer (uniform arg[4] consumed 0 properties)
        expect(layout.bindings[10].kind == RootSigBinding::Kind::SpilledSRV);
        expect(layout.bindings[10].element_stride == 112_u);
        // prop[11]: Usage::NONE 52B buffer
        expect(layout.bindings[11].kind == RootSigBinding::Kind::SpilledSRV);
        expect(layout.bindings[11].element_stride == 52_u);

        // Root params for spilled tables sit right after Direct params 0..4
        expect(layout.spilled_srv_root_param == 5_u);
        expect(layout.spilled_uav_root_param == 6_u);
        expect(layout.spilled_srv_count == 33_u);
        expect(layout.spilled_uav_count == 2_u);
    };

    "no_bindless_disambiguates_uniform_t0_vs_kernel_arg_t0"_test = [] {
        // Case A: NO bindless heaps, WITH uniform StructuredBuffer at t0/space0
        {
            vstd::vector<Property> props;
            props.emplace_back(Property{ShaderVariableType::ConstantValue, 4u, 0u, 1u});
            props.emplace_back(Property{ShaderVariableType::SamplerHeap, 1u, 0u, 16u});
            props.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, 0u, 1u});// _Global
            vstd::vector<SavedArgument> args;
            args.emplace_back(make_uniform_arg(4u));
            for (uint i = 0u; i < 32u; ++i) {
                props.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, i + 1u, 1u});
                args.emplace_back(make_buffer_arg(Usage::READ, 4u + i * 4u));
            }
            auto layout = lc::dx::build_root_sig_layout(props, args, false, false);
            expect(layout.spilled_tables);
            expect(layout.mapping_complete);
            expect(layout.kernel_arg_prop_offset == 3_u);
            expect(layout.bindings[2].kind == RootSigBinding::Kind::Direct);
            expect(layout.bindings[3].kind == RootSigBinding::Kind::SpilledSRV);
            expect(layout.bindings[3].element_stride == 4_u);
            expect(layout.bindings[4].element_stride == 8_u);
        }

        // Case B: NO bindless heaps, NO uniform buffer -> t0/space0 IS the first kernel arg
        {
            vstd::vector<Property> props;
            props.emplace_back(Property{ShaderVariableType::ConstantValue, 4u, 0u, 1u});
            props.emplace_back(Property{ShaderVariableType::SamplerHeap, 1u, 0u, 16u});
            vstd::vector<SavedArgument> args;
            for (uint i = 0u; i < 33u; ++i) {
                props.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, i, 1u});
                args.emplace_back(make_buffer_arg(Usage::READ, 8u + i * 4u));
            }
            auto layout = lc::dx::build_root_sig_layout(props, args, false, false);
            expect(layout.spilled_tables);
            expect(layout.mapping_complete);
            expect(layout.kernel_arg_prop_offset == 2_u);
            // First kernel SRV at t0/space0 stays Direct (2-DWORD Root SRV)
            expect(layout.bindings[2].kind == RootSigBinding::Kind::Direct);
            expect(layout.bindings[2].element_stride == 8_u);
            // Second kernel SRV at t1/space0 is SpilledSRV[0] with exact stride 12
            expect(layout.bindings[3].kind == RootSigBinding::Kind::SpilledSRV);
            expect(layout.bindings[3].index == 0_u);
            expect(layout.bindings[3].element_stride == 12_u);
        }
    };

    "single_textures_stay_direct_and_printers_map_at_tail"_test = [] {
        vstd::vector<Property> props;
        props.emplace_back(Property{ShaderVariableType::ConstantValue, 4u, 0u, 1u});
        props.emplace_back(Property{ShaderVariableType::SamplerHeap, 1u, 0u, 16u});
        props.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, 0u, 1u});// _Global
        vstd::vector<SavedArgument> args;
        args.emplace_back(make_uniform_arg(4u));

        // Single SRV texture and single UAV texture must stay Direct (never CopyDescriptorsSimple)
        props.emplace_back(Property{ShaderVariableType::SRVTextureHeap, 0u, 1u, 1u});
        args.emplace_back(make_texture_arg(Usage::READ));
        props.emplace_back(Property{ShaderVariableType::UAVTextureHeap, 0u, 0u, 1u});
        args.emplace_back(make_texture_arg(Usage::WRITE));

        for (uint i = 0u; i < 30u; ++i) {
            props.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, 2u + i, 1u});
            args.emplace_back(make_buffer_arg(Usage::READ, 16u));
        }
        // Tail printer properties (counter + raw byte buffer)
        props.emplace_back(Property{ShaderVariableType::RWStructuredBuffer, 0u, 1u, 1u});
        props.emplace_back(Property{ShaderVariableType::RWStructuredBuffer, 0u, 2u, 1u});

        auto layout = lc::dx::build_root_sig_layout(props, args, true, false);
        expect(layout.spilled_tables);
        expect(layout.mapping_complete);
        expect(layout.kernel_arg_prop_offset == 3_u);
        expect(layout.bindings[3].kind == RootSigBinding::Kind::Direct);
        expect(layout.bindings[4].kind == RootSigBinding::Kind::Direct);
        const size_t n = props.size();
        expect(layout.bindings[n - 2u].kind == RootSigBinding::Kind::SpilledUAV);
        expect(layout.bindings[n - 2u].element_stride == 4_u);
        expect(!layout.bindings[n - 2u].force_raw);
        expect(layout.bindings[n - 1u].kind == RootSigBinding::Kind::SpilledUAV);
        expect(layout.bindings[n - 1u].force_raw);
    };

    "real_hlsl_codegen_output_maps_cleanly_over_64_dwords"_test = [] {
        // Drive actual HLSL CodegenUtility::Codegen on a >64-DWORD compute kernel
        // with uniform struct, BindlessArray, ByteBuffer, Accel, Buffer<uint8_t>,
        // and >32 typed buffers (under emulate_8bit=true, matching SM 6.8 DX).
        Kernel1D kernel = [](
                              ByteBufferVar frame_buf,
                              BufferUInt count_buf,
                              AccelVar accel,
                              BindlessVar bdls,
                              BufferVar<TestPackedStruct> s0,
                              BufferVar<uint8_t> b8,
                              Var<TestUniformBlob> ublob,
                              BufferFloat b0, BufferFloat b1, BufferFloat b2, BufferFloat b3,
                              BufferFloat b4, BufferFloat b5, BufferFloat b6, BufferFloat b7,
                              BufferFloat b8f, BufferFloat b9, BufferFloat b10, BufferFloat b11,
                              BufferFloat b12, BufferFloat b13, BufferFloat b14, BufferFloat b15,
                              BufferFloat b16, BufferFloat b17, BufferFloat b18, BufferFloat b19,
                              BufferFloat b20, BufferFloat b21, BufferFloat b22, BufferFloat b23,
                              BufferFloat b24, BufferFloat b25, BufferFloat b26, BufferFloat b27,
                              BufferFloat b28, BufferFloat b29, BufferFloat b30, BufferFloat b31) noexcept {
            set_block_size(64u);
            auto idx = dispatch_x();
            auto hit = accel.intersect(make_ray(make_float3(0.f), make_float3(0.f, 0.f, 1.f), 0.f, 1e3f), {});
            auto v = bdls.buffer<uint>(0u).read(idx) +
                     s0.read(idx).tag +
                     cast<uint>(b8.read(idx)) +
                     ublob.a +
                     cast<uint>(hit.inst);
            auto sum = b0.read(idx) + b1.read(idx) + b2.read(idx) + b3.read(idx) +
                       b4.read(idx) + b5.read(idx) + b6.read(idx) + b7.read(idx) +
                       b8f.read(idx) + b9.read(idx) + b10.read(idx) + b11.read(idx) +
                       b12.read(idx) + b13.read(idx) + b14.read(idx) + b15.read(idx) +
                       b16.read(idx) + b17.read(idx) + b18.read(idx) + b19.read(idx) +
                       b20.read(idx) + b21.read(idx) + b22.read(idx) + b23.read(idx) +
                       b24.read(idx) + b25.read(idx) + b26.read(idx) + b27.read(idx) +
                       b28.read(idx) + b29.read(idx) + b30.read(idx) + b31.read(idx);
            frame_buf.write(idx * 4u, v + cast<uint>(sum));
            count_buf.atomic(0u).fetch_add(1u);
        };

        auto fn = kernel.function()->function();
        auto code = lc::hlsl::CodegenUtility{}.Codegen(
            fn, {}, 0u, /*isSpirV=*/false, /*noRegister=*/false,
            /*enable_debug_info=*/false, /*enable_fast_math=*/true,
            /*fallback_rtx=*/false, /*bf16_fallback=*/false,
            /*spirv_capabilities=*/{}, /*emulate_8bit=*/true);

        vstd::vector<SavedArgument> saved_args;
        for (auto &&var : fn.arguments()) {
            saved_args.emplace_back(fn, var);
        }

        auto layout = lc::dx::build_root_sig_layout(
            code.properties, saved_args, !code.printers.empty(), false);

        expect(layout.spilled_tables);
        expect(layout.mapping_complete);
        // Header: prop[0]=ConstVal, prop[1]=SamplerHeap, prop[2]=StructuredBuffer(_Global),
        // prop[3]=SRVBufferHeap (bindless buffer heap) -> kernel args start at prop[4]
        expect(layout.kernel_arg_prop_offset == 4_u);
        expect(code.properties[2].type == ShaderVariableType::StructuredBuffer);
        expect(code.properties[2].register_index == 0_u);
        expect(layout.bindings[2].kind == RootSigBinding::Kind::Direct);
        expect(layout.bindings[3].kind == RootSigBinding::Kind::Direct);

        // prop[4] = frame_buf (RWByteAddressBuffer): SpilledUAV, force_raw=true
        expect(layout.bindings[4].kind == RootSigBinding::Kind::SpilledUAV);
        expect(layout.bindings[4].force_raw);
        // prop[5] = count_buf (Buffer<uint>): SpilledUAV, stride=4, raw=false
        expect(layout.bindings[5].kind == RootSigBinding::Kind::SpilledUAV);
        expect(layout.bindings[5].element_stride == 4_u);
        expect(!layout.bindings[5].force_raw);
        // prop[6..7] = Accel (TopAccel + 64B _MeshInst)
        expect(layout.bindings[6].kind == RootSigBinding::Kind::SpilledSRV);
        expect(layout.bindings[6].element_stride == 0_u);
        expect(layout.bindings[7].kind == RootSigBinding::Kind::SpilledSRV);
        expect(layout.bindings[7].element_stride == 64_u);
        // prop[8] = BindlessArray (StructuredBuffer<uint>): stride=4
        expect(layout.bindings[8].kind == RootSigBinding::Kind::SpilledSRV);
        expect(layout.bindings[8].element_stride == 4_u);
        // prop[9] = s0 (Buffer<TestPackedStruct>): stride == sizeof(TestPackedStruct)
        expect(layout.bindings[9].kind == RootSigBinding::Kind::SpilledSRV);
        expect(layout.bindings[9].element_stride == static_cast<uint>(Type::of<TestPackedStruct>()->size()));
        // prop[10] = b8 (Buffer<uint8_t>): stride=1 (emulate_8bit lowers to raw ByteAddressBuffer on SM < 6.9)
        expect(layout.bindings[10].kind == RootSigBinding::Kind::SpilledSRV);
        expect(layout.bindings[10].element_stride == 1_u);
        // prop[11] = b0 (Buffer<float>, ublob consumed 0 properties): stride=4
        expect(layout.bindings[11].kind == RootSigBinding::Kind::SpilledSRV);
        expect(layout.bindings[11].element_stride == 4_u);
    };

    "emulate_8bit_packs_struct_and_compiles_sm68_dxil"_test = [&] {
        // Regression: SM < 6.9 D3D12 devices (e.g. inbox D3D12 SM 6.8) reject
        // native uint8_t/int8_t DXIL shaders. Under emulate_8bit=true, packed
        // 8-bit struct members and 1-byte buffers must lower cleanly to 32-bit
        // words + ByteAddressBuffer without setting use_8bit, and compile with
        // DXC to cs_6_8 DXIL.
        Kernel1D kernel = [](BufferVar<TestAttributeMap8Bit> attrs,
                             BufferVar<uint8_t> bytes_in,
                             BufferVar<uint8_t> bytes_out,
                             BufferUInt out) noexcept {
            set_block_size(32u);
            auto i = dispatch_x();
            auto m = attrs.read(i);
            auto b = bytes_in.read(i);
            m.flags = cast<uint8_t>(cast<uint>(m.flags) + 1u);
            attrs.write(i, m);
            bytes_out.write(i, cast<uint8_t>(cast<uint>(b) ^ cast<uint>(m.element)));
            out.write(i, m.id + m.offset + cast<uint>(m.type) + cast<uint>(cast<int>(m.signed_tag)));
        };
        auto fn = kernel.function()->function();
        auto code = lc::hlsl::CodegenUtility{}.Codegen(
            fn, {}, 0u, /*isSpirV=*/false, /*noRegister=*/false,
            /*enable_debug_info=*/false, /*enable_fast_math=*/true,
            /*fallback_rtx=*/false, /*bf16_fallback=*/false,
            /*spirv_capabilities=*/{}, /*emulate_8bit=*/true);
        expect(!code.use_8bit) << "emulate_8bit must not request SM 6.9 native 8-bit types";
        auto hlsl = code.result.view();
        expect(hlsl.find("ByteAddressBuffer") != luisa::string_view::npos);
        expect(hlsl.find("_lc_pack_get_u8") != luisa::string_view::npos);
        expect(hlsl.find("_lc_pack_set_u8") != luisa::string_view::npos);
        expect(hlsl.find("_lc_pack_get_i8") != luisa::string_view::npos);

        auto const runtime_directory = luisa::filesystem::path{argv[0]}.parent_path();
        lc::hlsl::ShaderCompiler compiler{runtime_directory, true};
        auto compiled = compiler.compile_compute(
            hlsl, /*optimize=*/true, /*shaderModel=*/68u, /*enableUnsafeMath=*/true,
            /*spirv=*/false, /*debug=*/false);
        expect(compiled.is_type_of<lc::hlsl::ComUniquePtr<IDxcBlob>>())
            << (compiled.is_type_of<vstd::string>() ? compiled.get<1>() : vstd::string{"unexpected DXC result"});
    };

    "emulate_8bit_aliased_struct_with_trailing_member_compiles_sm68_dxil"_test = [&] {
        // Exercises InitAsStructAliased + OriginToAliased/AliasedToOrigin when a
        // struct mixes a packed 8-bit word with an aliased bool3 and a trailing
        // non-packed member whose AST index skips over the packed members.
        Kernel1D kernel = [](BufferVar<TestAliasedPacked8Bit> buf,
                             BufferUInt out) noexcept {
            set_block_size(32u);
            auto i = dispatch_x();
            auto s = buf.read(i);
            s.a = cast<uint8_t>(cast<uint>(s.a) + 1u);
            s.tail = s.tail + s.id;
            buf.write(i, s);
            auto m = select(0u, 1u, s.mask.x & s.mask.y);
            out.write(i, s.tail + cast<uint>(s.b) + cast<uint>(cast<int>(s.c)) + m);
        };
        auto fn = kernel.function()->function();
        auto code = lc::hlsl::CodegenUtility{}.Codegen(
            fn, {}, 0u, /*isSpirV=*/false, /*noRegister=*/false,
            /*enable_debug_info=*/false, /*enable_fast_math=*/true,
            /*fallback_rtx=*/false, /*bf16_fallback=*/false,
            /*spirv_capabilities=*/{}, /*emulate_8bit=*/true);
        expect(!code.use_8bit);
        auto hlsl = code.result.view();
        expect(hlsl.find("uint v1;") != luisa::string_view::npos);
        expect(hlsl.find("uint v6;") != luisa::string_view::npos);

        auto const runtime_directory = luisa::filesystem::path{argv[0]}.parent_path();
        lc::hlsl::ShaderCompiler compiler{runtime_directory, true};
        auto compiled = compiler.compile_compute(
            hlsl, /*optimize=*/true, /*shaderModel=*/68u, /*enableUnsafeMath=*/true,
            /*spirv=*/false, /*debug=*/false);
        expect(compiled.is_type_of<lc::hlsl::ComUniquePtr<IDxcBlob>>())
            << (compiled.is_type_of<vstd::string>() ? compiled.get<1>() : vstd::string{"unexpected DXC result"});
    };

    "vector_scalar_bitcasts_compile_dxil_and_spirv"_test = [&] {
        // Regression for monster scene failure ("Vector to int unsupported."):
        // byte4/ubyte4 <-> uint/int, uint2/float2 <-> ulong/slong, and
        // half2/ushort2 <-> uint bitwise casts (as generated by SOA and Cycles
        // attributes) must compile cleanly for both SM 6.8 DXIL (emulate_8bit)
        // and SPIR-V.
        Kernel1D kernel = [](BufferUInt in_u32,
                             BufferULong in_u64,
                             BufferUInt out_u32,
                             BufferULong out_u64) noexcept {
            set_block_size(32u);
            auto i = dispatch_x();
            auto w = in_u32.read(i);
            auto q = in_u64.read(i);
            auto ub4 = as<ubyte4>(w);
            auto sb4 = as<byte4>(w);
            auto h2 = as<half2>(w);
            auto u2 = as<uint2>(q);
            auto f2 = as<float2>(q);
            auto w_back = as<uint>(ub4) ^ as<uint>(sb4) ^ as<uint>(h2);
            auto q_back = as<ulong>(u2) ^ as<ulong>(f2);
            out_u32.write(i, w_back);
            out_u64.write(i, q_back);
        };
        auto fn = kernel.function()->function();
        auto const runtime_directory = luisa::filesystem::path{argv[0]}.parent_path();
        lc::hlsl::ShaderCompiler compiler{runtime_directory, true};

        auto dx_code = lc::hlsl::CodegenUtility{}.Codegen(
            fn, {}, 0u, /*isSpirV=*/false, /*noRegister=*/false,
            /*enable_debug_info=*/false, /*enable_fast_math=*/true,
            /*fallback_rtx=*/false, /*bf16_fallback=*/false,
            /*spirv_capabilities=*/{}, /*emulate_8bit=*/true);
        expect(!dx_code.use_8bit);
        auto dx_compiled = compiler.compile_compute(
            dx_code.result.view(), /*optimize=*/true, /*shaderModel=*/68u,
            /*enableUnsafeMath=*/true, /*spirv=*/false, /*debug=*/false);
        expect(dx_compiled.is_type_of<lc::hlsl::ComUniquePtr<IDxcBlob>>())
            << (dx_compiled.is_type_of<vstd::string>() ? dx_compiled.get<1>() : vstd::string{"unexpected DXIL result"});

        auto spv_code = lc::hlsl::CodegenUtility{}.Codegen(
            fn, {}, 0u, /*isSpirV=*/true, /*noRegister=*/false,
            /*enable_debug_info=*/false, /*enable_fast_math=*/true,
            /*fallback_rtx=*/false, /*bf16_fallback=*/false,
            /*spirv_capabilities=*/{}, /*emulate_8bit=*/false);
        auto spv_compiled = compiler.compile_compute(
            spv_code.result.view(), /*optimize=*/true, /*shaderModel=*/66u,
            /*enableUnsafeMath=*/true, /*spirv=*/true, /*debug=*/false);
        expect(spv_compiled.is_type_of<lc::hlsl::ComUniquePtr<IDxcBlob>>())
            << (spv_compiled.is_type_of<vstd::string>() ? spv_compiled.get<1>() : vstd::string{"unexpected SPIR-V result"});
    };

    "hlsl_codegen_callable_order_is_deterministic"_test = [&] {
        // Two kernels with the same two callables allocated in opposite order
        // (simulating different ASLR heap addresses in _used_custom_callables)
        // must produce byte-identical HLSL so DX and VK-DXC hit disk caches.
        auto make_hlsl = [](bool reverse_alloc) {
            luisa::optional<Callable<uint(TestPackedStruct)>> fn_a;
            luisa::optional<Callable<uint(TestUniformBlob)>> fn_b;
            auto init_a = [&] {
                fn_a.emplace([](Var<TestPackedStruct> s) noexcept { return s.tag + s.a.x; });
            };
            auto init_b = [&] {
                fn_b.emplace([](Var<TestUniformBlob> u) noexcept { return u.a + u.b; });
            };
            if (reverse_alloc) {
                init_b();
                init_a();
            } else {
                init_a();
                init_b();
            }
            Kernel1D k = [&](BufferVar<TestPackedStruct> s,
                             BufferVar<TestUniformBlob> u,
                             BufferUInt out) noexcept {
                auto i = dispatch_x();
                out.write(i, (*fn_a)(s.read(i)) + (*fn_b)(u.read(i)));
            };
            auto code = lc::hlsl::CodegenUtility{}.Codegen(
                k.function()->function(), {}, 0u, /*isSpirV=*/false,
                /*noRegister=*/false, /*enable_debug_info=*/false,
                /*enable_fast_math=*/true);
            return luisa::string{code.result.view()};
        };
        expect(make_hlsl(false) == make_hlsl(true));
    };

    "hlsl_default_warp_size_emits_wavesize_for_dxil_only"_test = [&] {
        auto const runtime_directory = luisa::filesystem::path{argv[0]}.parent_path();
        lc::hlsl::ShaderCompiler compiler{runtime_directory, true};
        Kernel1D k_default = [](BufferUInt out) noexcept {
            set_block_size(32u);
            auto i = dispatch_x();
            out.write(i, warp_lane_count() + warp_lane_id());
        };
        Kernel1D k_explicit64 = [](BufferUInt out) noexcept {
            set_block_size(64u);
            set_warp_size(64u);
            auto i = dispatch_x();
            out.write(i, warp_lane_count() + warp_lane_id());
        };
        auto dx_default32 = lc::hlsl::CodegenUtility{}.Codegen(
            k_default.function()->function(), {}, 0u, /*isSpirV=*/false,
            /*noRegister=*/false, /*enable_debug_info=*/false,
            /*enable_fast_math=*/true, /*fallback_rtx=*/false,
            /*bf16_fallback=*/false, /*spirv_capabilities=*/{},
            /*emulate_8bit=*/false, /*default_warp_size=*/luisa::optional<uint8_t>{32u});
        expect(std::string_view{dx_default32.result.view()}.find("[WaveSize(32)]") !=
               std::string_view::npos);
        auto dx_compiled = compiler.compile_compute(
            dx_default32.result.view(), /*optimize=*/true, /*shaderModel=*/66u,
            /*enableUnsafeMath=*/true, /*spirv=*/false, /*debug=*/false);
        expect(dx_compiled.is_type_of<lc::hlsl::ComUniquePtr<IDxcBlob>>())
            << (dx_compiled.is_type_of<vstd::string>() ? dx_compiled.get<1>() : vstd::string{"unexpected DXIL WaveSize(32) result"});

        auto dx_override = lc::hlsl::CodegenUtility{}.Codegen(
            k_explicit64.function()->function(), {}, 0u, /*isSpirV=*/false,
            /*noRegister=*/false, /*enable_debug_info=*/false,
            /*enable_fast_math=*/true, /*fallback_rtx=*/false,
            /*bf16_fallback=*/false, /*spirv_capabilities=*/{},
            /*emulate_8bit=*/false, /*default_warp_size=*/luisa::optional<uint8_t>{32u});
        expect(std::string_view{dx_override.result.view()}.find("[WaveSize(64)]") !=
               std::string_view::npos);
        expect(std::string_view{dx_override.result.view()}.find("[WaveSize(32)]") ==
               std::string_view::npos);

        auto spv_default32 = lc::hlsl::CodegenUtility{}.Codegen(
            k_default.function()->function(), {}, 0u, /*isSpirV=*/true,
            /*noRegister=*/false, /*enable_debug_info=*/false,
            /*enable_fast_math=*/true, /*fallback_rtx=*/false,
            /*bf16_fallback=*/false, /*spirv_capabilities=*/{},
            /*emulate_8bit=*/false, /*default_warp_size=*/luisa::optional<uint8_t>{32u});
        expect(std::string_view{spv_default32.result.view()}.find("[WaveSize(") ==
               std::string_view::npos);
    };

    "hlsl_nested_empty_switch_inside_switch_with_default"_test = [&] {
        // Regression for HLSL codegen emitting `{ default: break; }` after the
        // closing `}` of an empty inner switch, which produced a duplicate
        // `default:` label inside an enclosing switch statement.
        auto const runtime_directory = luisa::filesystem::path{argv[0]}.parent_path();
        lc::hlsl::ShaderCompiler compiler{runtime_directory, true};
        Kernel1D kernel = [](BufferUInt in_a, BufferUInt in_b, BufferUInt out) noexcept {
            set_block_size(32u);
            auto i = dispatch_x();
            auto a = in_a.read(i);
            auto b = in_b.read(i);
            UInt acc = 0u;
            $switch(a) {
                $case(1u) {
                    $switch(b) {
                        $case(10u) {};
                        $default {};
                    };
                    acc = 11u;
                };
                $case(2u) {
                    $switch(b) {};
                    acc = 22u;
                };
                $default {
                    acc = 33u;
                };
            };
            out.write(i, acc);
        };
        auto fn = kernel.function()->function();
        auto dx_code = lc::hlsl::CodegenUtility{}.Codegen(
            fn, {}, 0u, /*isSpirV=*/false, /*noRegister=*/false,
            /*enable_debug_info=*/false, /*enable_fast_math=*/true);
        auto dx_compiled = compiler.compile_compute(
            dx_code.result.view(), /*optimize=*/true, /*shaderModel=*/66u,
            /*enableUnsafeMath=*/true, /*spirv=*/false, /*debug=*/false);
        expect(dx_compiled.is_type_of<lc::hlsl::ComUniquePtr<IDxcBlob>>())
            << (dx_compiled.is_type_of<vstd::string>() ? dx_compiled.get<1>() : vstd::string{"unexpected DXIL nested switch result"});

        auto spv_code = lc::hlsl::CodegenUtility{}.Codegen(
            fn, {}, 0u, /*isSpirV=*/true, /*noRegister=*/false,
            /*enable_debug_info=*/false, /*enable_fast_math=*/true);
        auto spv_compiled = compiler.compile_compute(
            spv_code.result.view(), /*optimize=*/true, /*shaderModel=*/66u,
            /*enableUnsafeMath=*/true, /*spirv=*/true, /*debug=*/false);
        expect(spv_compiled.is_type_of<lc::hlsl::ComUniquePtr<IDxcBlob>>())
            << (spv_compiled.is_type_of<vstd::string>() ? spv_compiled.get<1>() : vstd::string{"unexpected SPIR-V nested switch result"});
    };

    "hlsl_local_opt_prunes_const_switch_and_inlines_temps"_test = [&] {
        auto const runtime_directory = luisa::filesystem::path{argv[0]}.parent_path();
        lc::hlsl::ShaderCompiler compiler{runtime_directory, true};
        Kernel1D kernel = [](BufferFloat in_buf, BufferFloat out_buf) noexcept {
            set_block_size(32u);
            auto i = dispatch_x();
            Float x = in_buf.read(i);
            // Single-use temp chain should fold without hoisting entry allocas.
            Float t0 = x * 2.0f;
            Float t1 = t0 + 3.0f;
            Float acc = t1;
            // Constant switch with dead cases should prune dead cases.
            $switch(def(2u)) {
                $case(1u) {
                    acc = acc + 123456.0f;
                };
                $case(2u) {
                    acc = acc + 7.0f;
                };
                $case(3u) {
                    acc = acc + 654321.0f;
                };
            };
            // Constant false branch via SSA bool should be pruned.
            Bool dead_cond = (def(12u) == def(14u));
            $if(dead_cond) {
                acc = acc + 987654.0f;
            };
            // Constant switch with internal break inside an enclosing loop must
            // trap its break without exiting the outer loop.
            UInt k = 0u;
            $while(k < 4u) {
                $switch(def(1u)) {
                    $case(1u) {
                        $if(k == 2u) {
                            $break;
                        };
                        acc = acc + 1.0f;
                    };
                    $case(2u) {
                        acc = acc + 777777.0f;
                    };
                };
                k += 1u;
            };
            out_buf.write(i, acc);
        };
        auto fn = kernel.function()->function();
        auto dx_code = lc::hlsl::CodegenUtility{}.Codegen(
            fn, {}, 0u, /*isSpirV=*/false, /*noRegister=*/false,
            /*enable_debug_info=*/false, /*enable_fast_math=*/true);
        std::string_view src{dx_code.result.view().data(), dx_code.result.view().size()};
        // Dead constants from pruned switch cases and pruned if branch must not appear.
        expect(src.find("0x1.e240") == std::string_view::npos);// 123456.0f
        expect(src.find("0x1.3f8a") == std::string_view::npos);// 654321.0f
        expect(src.find("0x1.e240c") == std::string_view::npos);// 987654.0f
        expect(src.find("0x1.7bc2") == std::string_view::npos);// 777777.0f
        // Guard wrapper for internal break inside folded switch must be present.
        expect(src.find("switch(0u)") != std::string_view::npos);

        auto dx_compiled = compiler.compile_compute(
            dx_code.result.view(), /*optimize=*/true, /*shaderModel=*/66u,
            /*enableUnsafeMath=*/true, /*spirv=*/false, /*debug=*/false);
        expect(dx_compiled.is_type_of<lc::hlsl::ComUniquePtr<IDxcBlob>>())
            << (dx_compiled.is_type_of<vstd::string>() ? dx_compiled.get<1>() : vstd::string{"unexpected DXIL local-opt result"});

        auto spv_code = lc::hlsl::CodegenUtility{}.Codegen(
            fn, {}, 0u, /*isSpirV=*/true, /*noRegister=*/false,
            /*enable_debug_info=*/false, /*enable_fast_math=*/true);
        auto spv_compiled = compiler.compile_compute(
            spv_code.result.view(), /*optimize=*/true, /*shaderModel=*/66u,
            /*enableUnsafeMath=*/true, /*spirv=*/true, /*debug=*/false);
        expect(spv_compiled.is_type_of<lc::hlsl::ComUniquePtr<IDxcBlob>>())
            << (spv_compiled.is_type_of<vstd::string>() ? spv_compiled.get<1>() : vstd::string{"unexpected SPIR-V local-opt result"});
    };

    return 0;
}
