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
}// namespace detail

// Check if CBuffer has data (multiple ranges)
bool CodegenUtility::IsCBufferNonEmpty(std::initializer_list<vstd::IRange<Variable> *> fs) {
    for (auto f : fs) {
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
    vstd::StringBuilder &result) {
    result << "struct _Args{\n"sv;
    size_t align = 0;
    size_t size = 0;
    size_t struct_size = 0;
    for (auto &&f : fs) {
        size_t size_cache = 0;
        Type const *last_type = nullptr;
        for (auto &&i : *f) {
            if (!detail::IsCBuffer(i.tag())) continue;
            size_cache++;
            StructGenerator::ProvideAlignVariable(last_type, i.type()->alignment(), align, struct_size, result);
            if (last_type && (StructGenerator::half_type_adjacent_with_bool(last_type, i.type()) ||
                              StructGenerator::half_type_adjacent_with_bool(i.type(), last_type))) [[unlikely]] {
                LUISA_ERROR("HLSL do not support 16-bit variables adjacent with bool");
            }
            last_type = i.type();
            if (opt->isSpirv && i.type()->tag() == Type::Tag::BOOL) {
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
        size += size_cache;
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
