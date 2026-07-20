// Tests for the HLSL debug-validation producer and callable ABI.

#include "ut/ut.hpp"

#include <luisa/core/stl/format.h>
#include <luisa/dsl/sugar.h>

#include "hlsl_codegen.h"
#include "atomic_codegen_policy.h"
#include "shader_compiler.h"

#include <filesystem>
#include <string_view>

#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] bool contains(
    luisa::string_view text,
    std::string_view needle) noexcept {
    return text.find(needle) != luisa::string_view::npos;
}

[[nodiscard]] size_t count_substring(
    luisa::string_view text,
    std::string_view needle) noexcept {
    size_t count = 0u;
    for (auto offset = size_t{0u};;) {
        auto position = text.find(needle, offset);
        if (position == luisa::string_view::npos) { break; }
        ++count;
        offset = position + needle.size();
    }
    return count;
}

[[nodiscard]] bool is_identifier_character(char c) noexcept {
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

[[nodiscard]] bool calls_have_arity(
    luisa::string_view text,
    std::string_view name,
    size_t expected_arity,
    size_t expected_count,
    bool exact_identifier = true) noexcept {
    auto call_count = size_t{0u};
    for (auto offset = size_t{0u};;) {
        auto position = text.find(name, offset);
        if (position == luisa::string_view::npos) { break; }
        offset = position + name.size();
        if (exact_identifier &&
            ((position != 0u &&
              is_identifier_character(text[position - 1u])) ||
             (offset < text.size() &&
              is_identifier_character(text[offset])))) {
            continue;
        }
        auto open = text.find('(', offset);
        if (open == luisa::string_view::npos) { return false; }
        auto depth = size_t{1u};
        auto arity = size_t{1u};
        auto close = luisa::string_view::npos;
        for (auto i = open + 1u; i < text.size(); ++i) {
            switch (text[i]) {
                case '(': ++depth; break;
                case ')':
                    if (--depth == 0u) { close = i; }
                    break;
                case ',':
                    if (depth == 1u) { ++arity; }
                    break;
                default: break;
            }
            if (close != luisa::string_view::npos) { break; }
        }
        if (close == luisa::string_view::npos || arity != expected_arity) {
            return false;
        }
        ++call_count;
        offset = close + 1u;
    }
    return call_count == expected_count;
}

[[nodiscard]] luisa::string_view generated_program(
    lc::hlsl::CodegenResult const &codegen) noexcept {
    auto text = codegen.result.view();
    auto offset = text.find("struct _Args{");
    return offset == luisa::string_view::npos ?
               luisa::string_view{} :
               text.substr(offset);
}

[[nodiscard]] size_t property_count(
    lc::hlsl::CodegenResult const &codegen,
    lc::hlsl::ShaderVariableType type) noexcept {
    auto count = size_t{0u};
    for (auto property : codegen.properties) {
        if (property.type == type) { ++count; }
    }
    return count;
}

[[nodiscard]] size_t property_index(
    lc::hlsl::CodegenResult const &codegen,
    lc::hlsl::ShaderVariableType type) noexcept {
    for (auto i = size_t{0u}; i < codegen.properties.size(); ++i) {
        if (codegen.properties[i].type == type) { return i; }
    }
    return codegen.properties.size();
}

struct DxcSpirvAtomicFacts {
    bool compiled{false};
    bool validated{false};
    size_t compare_exchange_count{0u};
    size_t float_add_count{0u};
    size_t integer_add_count{0u};
    vstd::string error;
};

[[nodiscard]] DxcSpirvAtomicFacts compile_and_inspect_dxc_spirv(
    luisa::string_view source,
    const std::filesystem::path &runtime_directory) {
    DxcSpirvAtomicFacts facts;
    lc::hlsl::ShaderCompiler compiler{runtime_directory, true};
    auto compiled = compiler.compile_compute(
        source, true, 65u, true, true, false);
    compiled.multi_visit(
        [&](const lc::hlsl::ComUniquePtr<IDxcBlob> &blob) {
            facts.compiled = true;
            auto byte_size = blob->GetBufferSize();
            if (byte_size % sizeof(uint32_t) != 0u) {
                facts.error = "DXC returned a non-word-aligned SPIR-V blob";
                return;
            }
            auto *words = static_cast<const uint32_t *>(
                blob->GetBufferPointer());
            auto word_count = byte_size / sizeof(uint32_t);
            spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
            facts.validated = tools.Validate(words, word_count);
            for (auto offset = size_t{5u}; offset < word_count;) {
                auto instruction_word_count =
                    static_cast<size_t>(words[offset] >> 16u);
                if (instruction_word_count == 0u ||
                    instruction_word_count > word_count - offset) {
                    facts.error = "DXC returned malformed SPIR-V";
                    break;
                }
                auto opcode = static_cast<spv::Op>(
                    words[offset] & 0xffffu);
                if (opcode == spv::Op::OpAtomicCompareExchange) {
                    ++facts.compare_exchange_count;
                } else if (opcode == spv::Op::OpAtomicFAddEXT) {
                    ++facts.float_add_count;
                } else if (opcode == spv::Op::OpAtomicIAdd) {
                    ++facts.integer_add_count;
                }
                offset += instruction_word_count;
            }
        },
        [&](const vstd::string &error) {
            facts.error = error;
        });
    return facts;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "hlsl_codegen_result_defaults_are_deterministic"_test = [] {
        lc::hlsl::CodegenResult result{};
        expect(!result.useTex2DBindless);
        expect(!result.useTex3DBindless);
        expect(!result.useBufferBindless);
        expect(!result.use_8bit);
        expect(eq(result.validation_count, 0u));
        expect(eq(result.immutableHeaderSize, 0u));
    };

    "hlsl_float_atomic_codegen_respects_spirv_boundary"_test = [argv] {
        constexpr CallOp float_ops[]{
            CallOp::ATOMIC_EXCHANGE,
            CallOp::ATOMIC_COMPARE_EXCHANGE,
            CallOp::ATOMIC_FETCH_ADD,
            CallOp::ATOMIC_FETCH_SUB,
            CallOp::ATOMIC_FETCH_MIN,
            CallOp::ATOMIC_FETCH_MAX,
        };
        for (auto op : float_ops) {
            expect(
                lc::hlsl::plan_hlsl_atomic_lowering(op, true, true) ==
                lc::hlsl::HlslAtomicLowering::UNSUPPORTED);
        }
        expect(
            lc::hlsl::plan_hlsl_atomic_lowering(
                CallOp::ATOMIC_FETCH_ADD, false, true) ==
            lc::hlsl::HlslAtomicLowering::NATIVE);

        Kernel1D buffer_kernel = [](BufferFloat values,
                                    BufferFloat old_values) noexcept {
            old_values.write(
                0u, values.atomic(0u).fetch_add(1.0f));
        };
        auto function = buffer_kernel.function()->function();
        auto dx_software = lc::hlsl::CodegenUtility{}.Codegen(
            function, {}, 0u, false, false, false);
        auto software_program = dx_software.result.view();
        expect(contains(
            software_program,
            "InterlockedCompareExchangeFloatBitwise("));
        expect(contains(
            software_program,
            "if(asuint(old)==asuint(r))return old;"))
            << "CAS success must compare the exact float bit patterns";
        expect(contains(software_program, "old=r;"))
            << "CAS failure must reuse the returned bits instead of reloading";
        expect(!contains(software_program, "InterlockedAdd("));

        Kernel1D integer_kernel = [](BufferUInt values,
                                     BufferUInt old_values) noexcept {
            old_values.write(
                0u, values.atomic(0u).fetch_add(1u));
        };
        auto integer_spirv = lc::hlsl::CodegenUtility{}.Codegen(
            integer_kernel.function()->function(), {}, 0u,
            true, false, false);
        auto integer_program = integer_spirv.result.view();
        expect(contains(integer_program, "InterlockedAdd("));
        auto dxc = compile_and_inspect_dxc_spirv(
            integer_program,
            std::filesystem::path{argv[0]}.parent_path());
        expect(dxc.compiled) << dxc.error;
        expect(dxc.validated) << dxc.error;
        expect(dxc.integer_add_count > 0u);
        expect(eq(dxc.float_add_count, 0u));
        expect(eq(dxc.compare_exchange_count, 0u));
    };

    "hlsl_debug_validation_is_forwarded_through_callables"_test = [] {
        Callable inner = [](BufferFloat buffer,
                            BindlessVar bindless,
                            UInt index) noexcept {
            auto direct = buffer.read(index);
            auto untyped =
                bindless.buffer<float>(0u).read(index);
            auto uniform =
                bindless.buffer<float>(1u, false, true).read(index);
            auto typed =
                bindless.buffer<float>(2u, true, false).read(index);
            auto typed_uniform =
                bindless.buffer<float>(3u, true, true).read(index);
            auto untyped_byte =
                bindless.byte_buffer(4u).read<float>(0u);
            auto uniform_byte =
                bindless.byte_buffer(5u, false, true).read<float>(0u);
            auto typed_byte =
                bindless.byte_buffer(6u, true, false).read<float>(0u);
            auto typed_uniform_byte =
                bindless.byte_buffer(7u, true, true).read<float>(0u);
            return direct + untyped + uniform + typed + typed_uniform +
                   untyped_byte + uniform_byte + typed_byte +
                   typed_uniform_byte;
        };
        inner.set_name("validation_inner_callable_abi");

        Callable outer = [&inner](BufferFloat buffer,
                                  BindlessVar bindless,
                                  UInt index) noexcept {
            auto value = inner(buffer, bindless, index);
            buffer.write(index, value);
            return value;
        };
        outer.set_name("validation_outer_callable_abi");

        Kernel1D kernel = [&outer](BufferFloat buffer,
                                   BindlessVar bindless,
                                   UInt index_seed) noexcept {
            auto index = dispatch_x() + index_seed;
            buffer.write(index, outer(buffer, bindless, index));
        };
        auto function = kernel.function()->function();

        auto debug = lc::hlsl::CodegenUtility{}.Codegen(
            function, {}, 0u, true, false, true);
        auto debug_program = generated_program(debug);
        expect(!debug_program.empty());
        expect(eq(debug.validation_count, 2u));
        expect(contains(debug_program, "uint _validate_0;"));
        expect(contains(debug_program, "uint _validate_1;"));
        expect(!contains(debug_program, "uint _validate_2;"));
        expect(contains(debug_program, "_Global[0]._validate_0"));
        expect(contains(debug_program, "_Global[0]._validate_1"));
        expect(!contains(debug.result.view(), "_validate_##"));

        auto inner_arguments = inner.function().arguments();
        auto outer_arguments = outer.function().arguments();
        for (auto variable : {inner_arguments[0], inner_arguments[1],
                              outer_arguments[0], outer_arguments[1]}) {
            auto name = luisa::format(
                "_validation_bound_{}", variable.uid());
            expect(contains(debug_program, name));
        }
        expect(count_substring(debug_program, "_validation_bound_") >= 8u);

        expect(calls_have_arity(
            debug_program, "validation_inner_callable_abi",
            5u, 2u, false));
        expect(calls_have_arity(
            debug_program, "validation_outer_callable_abi",
            5u, 2u, false));
        expect(calls_have_arity(debug_program, "_bfread", 3u, 1u));
        expect(calls_have_arity(debug_program, "_bfwrite", 4u, 2u));
        expect(calls_have_arity(debug_program, "_READ_BUFFER", 7u, 1u));
        expect(calls_have_arity(
            debug_program, "_uniform_READ_BUFFER", 7u, 1u));
        expect(calls_have_arity(
            debug_program, "_typed_READ_BUFFER", 7u, 1u));
        expect(calls_have_arity(
            debug_program, "_typed_uniform_READ_BUFFER", 7u, 1u));
        expect(calls_have_arity(
            debug_program, "_READ_BUFFER_BYTES", 6u, 1u));
        expect(calls_have_arity(
            debug_program, "_uniform_READ_BUFFER_BYTES", 6u, 1u));
        expect(calls_have_arity(
            debug_program, "_typed_READ_BUFFER_BYTES", 6u, 1u));
        expect(calls_have_arity(
            debug_program, "_typed_uniform_READ_BUFFER_BYTES", 6u, 1u));

        auto release = lc::hlsl::CodegenUtility{}.Codegen(
            function, {}, 0u, true, false, false);
        auto release_program = generated_program(release);
        expect(!release_program.empty());
        expect(eq(release.validation_count, 0u));
        expect(!contains(release_program, "uint _validate_0;"));
        expect(!contains(release_program, "_validation_bound_"));
        expect(calls_have_arity(
            release_program, "validation_inner_callable_abi",
            3u, 2u, false));
        expect(calls_have_arity(
            release_program, "validation_outer_callable_abi",
            3u, 2u, false));
        expect(calls_have_arity(release_program, "_bfread", 2u, 1u));
        expect(calls_have_arity(release_program, "_bfwrite", 3u, 2u));
        expect(calls_have_arity(release_program, "_READ_BUFFER", 6u, 1u));
        expect(calls_have_arity(
            release_program, "_uniform_READ_BUFFER", 6u, 1u));
        expect(calls_have_arity(
            release_program, "_typed_READ_BUFFER", 6u, 1u));
        expect(calls_have_arity(
            release_program, "_typed_uniform_READ_BUFFER", 6u, 1u));
        expect(calls_have_arity(
            release_program, "_READ_BUFFER_BYTES", 5u, 1u));
        expect(calls_have_arity(
            release_program, "_uniform_READ_BUFFER_BYTES", 5u, 1u));
        expect(calls_have_arity(
            release_program, "_typed_READ_BUFFER_BYTES", 5u, 1u));
        expect(calls_have_arity(
            release_program, "_typed_uniform_READ_BUFFER_BYTES", 5u, 1u));

        auto fallback_debug = lc::hlsl::CodegenUtility{}.Codegen(
            function, {}, 0u, false, true, true);
        auto fallback_program = generated_program(fallback_debug);
        expect(!fallback_program.empty());
        expect(eq(fallback_debug.validation_count, 2u));
        expect(calls_have_arity(
            fallback_program, "validation_inner_callable_abi",
            5u, 2u, false));
        expect(calls_have_arity(fallback_program, "_bfread", 3u, 1u));
        expect(calls_have_arity(
            fallback_program, "_READ_BUFFER", 7u, 1u));
        expect(calls_have_arity(
            fallback_program, "_uniform_READ_BUFFER", 7u, 1u));
        expect(calls_have_arity(
            fallback_program, "_typed_READ_BUFFER", 7u, 1u));
        expect(calls_have_arity(
            fallback_program, "_typed_uniform_READ_BUFFER", 7u, 1u));
        expect(calls_have_arity(
            fallback_program, "_READ_BUFFER_BYTES", 6u, 1u));
        expect(calls_have_arity(
            fallback_program, "_uniform_READ_BUFFER_BYTES", 6u, 1u));
        expect(calls_have_arity(
            fallback_program, "_typed_READ_BUFFER_BYTES", 6u, 1u));
        expect(calls_have_arity(
            fallback_program, "_typed_uniform_READ_BUFFER_BYTES", 6u, 1u));
    };

    "hlsl_spirv_read_write_texture_uses_split_callable_views"_test = [] {
        Callable sample_and_write = [](ImageFloat image,
                                       Float2 uv,
                                       UInt2 coord) noexcept {
            auto &builder =
                *luisa::compute::detail::FunctionBuilder::current();
            auto level = builder.literal(Type::of<float>(), 0.0f);
            auto filter = builder.literal(Type::of<uint>(), 0u);
            auto address = builder.literal(Type::of<uint>(), 0u);
            auto sampled_expression = builder.call(
                Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE_LEVEL,
                {image.expression(), uv.expression(), level,
                 filter, address});
            auto sampled = def<float4>(sampled_expression);
            image.write(coord, sampled);
            return sampled;
        };
        sample_and_write.set_name("split_texture_callable_abi");

        Kernel2D kernel = [&sample_and_write](
                              ImageFloat image,
                              BufferFloat4 output) noexcept {
            auto sampled = sample_and_write(
                image, make_float2(0.5f), dispatch_id().xy());
            output.write(dispatch_x(), sampled);
        };
        auto function = kernel.function()->function();
        auto codegen = lc::hlsl::CodegenUtility{}.Codegen(
            function, {}, 0u, true, false, false);

        using lc::hlsl::ShaderVariableType;
        expect(eq(property_count(
                      codegen, ShaderVariableType::SRVTextureHeap),
                  1u));
        expect(eq(property_count(
                      codegen, ShaderVariableType::UAVTextureHeap),
                  1u));
        auto srv_index = property_index(
            codegen, ShaderVariableType::SRVTextureHeap);
        auto uav_index = property_index(
            codegen, ShaderVariableType::UAVTextureHeap);
        expect(srv_index < codegen.properties.size() &&
               uav_index == srv_index + 1u);
        if (srv_index < codegen.properties.size() &&
            uav_index < codegen.properties.size()) {
            auto &srv_property = codegen.properties[srv_index];
            auto &uav_property = codegen.properties[uav_index];
            expect(eq(srv_property.space_index, 0u));
            expect(eq(uav_property.space_index, 0u));
            expect(eq(srv_property.array_size, 1u));
            expect(eq(uav_property.array_size, 1u));
            expect(eq(uav_property.register_index,
                      srv_property.register_index + 1u));
        }

        auto root_texture = function.arguments()[0];
        auto root_name = luisa::format("_t{}", root_texture.uid());
        auto callable_texture = sample_and_write.function().arguments()[0];
        auto callable_name =
            luisa::format("_t{}", callable_texture.uid());
        auto hlsl = codegen.result.view();
        if (srv_index < codegen.properties.size() &&
            uav_index < codegen.properties.size()) {
            auto srv_declaration = luisa::format(
                "Texture2D<float4> {}:register(t{});", root_name,
                codegen.properties[srv_index].register_index);
            auto uav_declaration = luisa::format(
                "RWTexture2D<float4> {}_rw:register(u{});", root_name,
                codegen.properties[uav_index].register_index);
            auto srv_declaration_index = hlsl.find(srv_declaration);
            auto uav_declaration_index = hlsl.find(uav_declaration);
            expect(srv_declaration_index != luisa::string_view::npos);
            expect(uav_declaration_index != luisa::string_view::npos);
            expect(srv_declaration_index < uav_declaration_index);
        }
        expect(contains(
            hlsl, luisa::format(
                      "_SmptxLevel({},", callable_name)));
        expect(contains(
            hlsl, luisa::format(
                      "_Writetx({}_rw,", callable_name)));
        expect(contains(
            hlsl, luisa::format(
                      "({},{}_rw,", root_name, root_name)));
        expect(calls_have_arity(
            hlsl, "split_texture_callable_abi", 4u, 2u, false));
    };

    "hlsl_byte_buffer_validation_macro_arities_are_consistent"_test = [] {
        Kernel1D kernel = [](ByteBufferVar buffer, UInt byte_offset) noexcept {
            auto matrix = buffer.read<float2x2>(byte_offset);
            buffer.write(byte_offset, matrix);
            auto vector = buffer.read<float3>(byte_offset + 32u);
            buffer.write(byte_offset + 32u, vector);
            auto value = buffer.read<uint>(byte_offset + 64u);
            buffer.write(byte_offset + 64u, value);
            auto volatile_value =
                buffer.volatile_read<uint>(byte_offset + 80u);
            buffer.volatile_write(byte_offset + 80u, volatile_value);
        };
        auto function = kernel.function()->function();

        auto debug = lc::hlsl::CodegenUtility{}.Codegen(
            function, {}, 0u, true, false, true);
        auto debug_program = generated_program(debug);
        expect(!debug_program.empty());
        expect(eq(debug.validation_count, 1u));
        expect(calls_have_arity(
            debug_program, "_bytebfreadMat", 4u, 1u));
        expect(calls_have_arity(
            debug_program, "_bytebfwriteMat", 5u, 1u));
        expect(calls_have_arity(
            debug_program, "_bytebfreadVec3", 4u, 1u));
        expect(calls_have_arity(
            debug_program, "_bytebfwriteVec3", 5u, 1u));
        expect(calls_have_arity(debug_program, "_bytebfread", 4u, 1u));
        expect(calls_have_arity(debug_program, "_bytebfwrite", 4u, 1u));
        expect(calls_have_arity(
            debug_program, "_volatile_bytebfread", 2u, 1u));
        expect(calls_have_arity(
            debug_program, "_volatile_bytebfwrite", 3u, 1u));

        auto release = lc::hlsl::CodegenUtility{}.Codegen(
            function, {}, 0u, true, false, false);
        auto release_program = generated_program(release);
        expect(!release_program.empty());
        expect(eq(release.validation_count, 0u));
        expect(calls_have_arity(
            release_program, "_bytebfreadMat", 3u, 1u));
        expect(calls_have_arity(
            release_program, "_bytebfwriteMat", 4u, 1u));
        expect(calls_have_arity(
            release_program, "_bytebfreadVec3", 3u, 1u));
        expect(calls_have_arity(
            release_program, "_bytebfwriteVec3", 4u, 1u));
        expect(calls_have_arity(release_program, "_bytebfread", 3u, 1u));
        expect(calls_have_arity(release_program, "_bytebfwrite", 3u, 1u));
        expect(calls_have_arity(
            release_program, "_volatile_bytebfread", 2u, 1u));
        expect(calls_have_arity(
            release_program, "_volatile_bytebfwrite", 3u, 1u));
    };

    "hlsl_structured_buffer_validation_macro_arities_are_consistent"_test = [] {
        Kernel1D kernel = [](BufferFloat scalar,
                             BufferFloat3 vector,
                             BufferFloat2x2 matrix,
                             UInt index) noexcept {
            scalar.write(index, scalar.read(index));
            vector.write(index, vector.read(index));
            matrix.write(index, matrix.read(index));
        };
        auto function = kernel.function()->function();

        auto check_debug = [](luisa::string_view program) noexcept {
            return calls_have_arity(program, "_bfread", 3u, 1u) &&
                   calls_have_arity(program, "_bfwrite", 4u, 1u) &&
                   calls_have_arity(program, "_bfreadVec3", 3u, 1u) &&
                   calls_have_arity(program, "_bfwriteVec3", 5u, 1u) &&
                   calls_have_arity(program, "_bfreadMat", 3u, 1u) &&
                   calls_have_arity(program, "_bfwriteMat", 4u, 1u);
        };
        auto check_release = [](luisa::string_view program) noexcept {
            return calls_have_arity(program, "_bfread", 2u, 1u) &&
                   calls_have_arity(program, "_bfwrite", 3u, 1u) &&
                   calls_have_arity(program, "_bfreadVec3", 2u, 1u) &&
                   calls_have_arity(program, "_bfwriteVec3", 4u, 1u) &&
                   calls_have_arity(program, "_bfreadMat", 2u, 1u) &&
                   calls_have_arity(program, "_bfwriteMat", 3u, 1u);
        };

        auto debug = lc::hlsl::CodegenUtility{}.Codegen(
            function, {}, 0u, true, false, true);
        expect(eq(debug.validation_count, 3u));
        expect(check_debug(generated_program(debug)));

        auto release = lc::hlsl::CodegenUtility{}.Codegen(
            function, {}, 0u, true, false, false);
        expect(eq(release.validation_count, 0u));
        expect(check_release(generated_program(release)));

        auto fallback_debug = lc::hlsl::CodegenUtility{}.Codegen(
            function, {}, 0u, false, true, true);
        expect(eq(fallback_debug.validation_count, 3u));
        expect(check_debug(generated_program(fallback_debug)));
    };
}
