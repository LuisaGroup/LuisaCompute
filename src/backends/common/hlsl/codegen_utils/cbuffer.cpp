// CBuffer (Constant Buffer) Management

#include "../hlsl_codegen.h"
#include "../struct_generator.h"
#include "../codegen_stack_data.h"

namespace lc::hlsl {

namespace detail {
bool IsCBuffer(Variable::Tag t) {
    switch (t) {
        case Variable::Tag::BUFFER:
        case Variable::Tag::TEXTURE:
        case Variable::Tag::BINDLESS_ARRAY:
        case Variable::Tag::ACCEL:
        case Variable::Tag::THREAD_ID:
        case Variable::Tag::BLOCK_ID:
        case Variable::Tag::DISPATCH_ID:
        case Variable::Tag::DISPATCH_SIZE:
        case Variable::Tag::KERNEL_ID:
        case Variable::Tag::RASTER_BARYCENTRICS:
        case Variable::Tag::RASTER_OBJECT_ID:
            return false;
        default:
            return true;
    }
}
// The resource can be validate index out or range
bool IsValidateResource(Variable::Tag t) {
    switch (t) {
        case Variable::Tag::BUFFER:
        case Variable::Tag::BINDLESS_ARRAY:
            return true;
        default:
            return false;
    }
}
}// namespace detail

// Check if CBuffer has data (multiple ranges)
bool CodegenUtility::IsCBufferNonEmpty(std::initializer_list<vstd::IRange<Variable> *> fs) {
    for (auto &&f : fs) {
        for (auto &&i : *f) {
            if (detail::IsCBuffer(i.tag())) {
                return true;
            }
        }
    }
    return false;
}

// Check if CBuffer has data (single function)
bool CodegenUtility::IsCBufferNonEmpty(Function f) {
    for (auto &&i : f.arguments()) {
        if (detail::IsCBuffer(i.tag())) {
            return true;
        }
    }
    return false;
}

// Generate CBuffer declaration
void CodegenUtility::GenerateCBuffer(
    std::initializer_list<vstd::IRange<Variable> *> fs,
    vstd::StringBuilder &result,
    bool generate_debug_info,
    uint &validation_count,
    vstd::span<const uint64_t> func_hashes) {
    validation_count = 0;
    result << "struct _Args{\n"sv;
    size_t align = 0;
    size_t size = 0;
    size_t struct_size = 0;
    size_t func_idx = 0;
    for (auto &&f : fs) {
        size_t size_cache = 0;
        Type const *last_type = nullptr;
        for (auto &&i : *f) {
            if (!detail::IsCBuffer(i.tag())) {
                if (generate_debug_info && detail::IsValidateResource(i.tag())) {
                    if (func_idx < func_hashes.size()) {
                        uint64_t func_hash = func_hashes[func_idx];
                        opt->validate_index_map[CodegenStackData::ValidateKey{func_hash, i.uid()}] = validation_count;
                    }
                    ++validation_count;
                }
            } else {
                size_cache++;
                StructGenerator::ProvideAlignVariable(last_type, i.type()->alignment(), align, struct_size, result);
                if (last_type && (StructGenerator::half_type_adjacent_with_bool(last_type, i.type()) ||
                                  StructGenerator::half_type_adjacent_with_bool(i.type(), last_type))) [[unlikely]] {
                    LUISA_ERROR("HLSL do not support 16-bit variables adjacent with bool");
                }
                last_type = i.type();
                // Note: BOOL scalar kernel arguments go through this cbuffer code path.
                // They are emitted as `int l<uid>:8` bitfields to match C++ bool size (1 byte).
                // Verified by test with Var<bool> kernel arguments in
                // src/tests/unit/dsl/test_dsl.cpp (bool+ite test + multi_bool test).
                //
                // Bool vectors (bool2/3/4) do NOT need the same treatment because:
                // - GetTypeName with local_var=true emits boolN (e.g. bool2)
                // - In HLSL/DXIL, boolN maps to <N x i32>, not i8, so no I8 error.
                // - Verified by test with Var<bool2> kernel argument in the same file.
                if (i.type()->tag() == Type::Tag::BOOL) {
                    result << "int";
                } else
                    GetTypeName(*i.type(), result, Usage::READ, true);
                if (opt->isSpirv && i.type()->tag() != Type::Tag::BOOL && i.type()->alignment() < 4) [[unlikely]] {
                    LUISA_ERROR("Member less than 4-byte can not be argument in SPIRV.");
                }
                struct_size += i.type()->size();
                result << " l" << vstd::to_string(i.uid() + size);
                if (i.type()->tag() == Type::Tag::BOOL) {
                    result << ":8"sv;
                }
                result << ";\n"sv;
                if (i.type()->is_vector() && i.type()->dimension() == 3) {
                    GetTypeName(*i.type()->element(), result, Usage::READ, true);
                    result << " _a"sv;
                    vstd::to_string(align, result);
                    result << ";\n"sv;
                    ++align;
                }
            }
        }
        size += size_cache;
        ++func_idx;
    }
    // generate _validate_* variable in kernel
    if (generate_debug_info) {
        for (auto i : vstd::range(validation_count)) {
            result << "uint _validate_" << luisa::format("{}", i) << ";\n";
        }
    }
    if (opt->noRegister) {
        result << R"(};
StructuredBuffer<_Args> _Global;
)"sv;
    } else {
        result << R"(};
StructuredBuffer<_Args> _Global:register(t0);
)"sv;
    }
}

}// namespace lc::hlsl
