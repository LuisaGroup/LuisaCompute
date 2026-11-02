#pragma once
#include <luisa/ast/function.h>
#include <luisa/ast/op.h>
#include <luisa/ast/type.h>

namespace lc::vk::detail {

// Whether an AST type carries a BF16 payload anywhere.
//
// The native XIR-to-SPIR-V route can only express BF16 with the device's
// `shaderBFloat16Type` (VK_KHR_shader_bfloat16) - the SPIR-V codegen refuses a
// BFloat16 type without it (spirv_codegen/type.cpp).  A device without the
// feature therefore has to compile a BF16 kernel through the compatibility
// HLSL route, where BF16 is emulated in software by
// src/backends/common/hlsl/builtin/bfloat16_fallback.bytes.
[[nodiscard]] inline bool type_uses_bfloat16(Type const *type) noexcept {
    if (type == nullptr) { return false; }
    switch (type->tag()) {
        case Type::Tag::BFLOAT16:
            return true;
        case Type::Tag::VECTOR:
        case Type::Tag::MATRIX:
        case Type::Tag::BUFFER:
        case Type::Tag::ARRAY:
        case Type::Tag::TEXTURE:
        case Type::Tag::COOPERATIVE_VECTOR:
            return type_uses_bfloat16(type->element());
        case Type::Tag::STRUCTURE:
            for (auto member : type->members()) {
                if (type_uses_bfloat16(member)) { return true; }
            }
            return false;
        default:
            return false;
    }
}

// Whether the kernel (or a callable it uses) mentions BF16 anywhere: in its
// arguments, locals, shared variables, constants, or in a BF16 vector
// construction.  A miss is fail-closed: the native SPIR-V route then refuses the
// shader with an explicit "requires shaderBFloat16Type" error instead of
// silently emitting something else.
[[nodiscard]] inline bool function_uses_bfloat16(Function kernel) noexcept {
    if (!kernel) { return false; }
    auto scan_variables = [](luisa::span<const Variable> variables) noexcept {
        for (auto &&v : variables) {
            if (type_uses_bfloat16(v.type())) { return true; }
        }
        return false;
    };
    if (scan_variables(kernel.arguments()) ||
        scan_variables(kernel.unbound_arguments()) ||
        scan_variables(kernel.local_variables()) ||
        scan_variables(kernel.shared_variables()) ||
        scan_variables(kernel.builtin_variables())) {
        return true;
    }
    for (auto &&constant : kernel.constants()) {
        if (type_uses_bfloat16(constant.type())) { return true; }
    }
    for (auto &&callable : kernel.custom_callables()) {
        if (function_uses_bfloat16(Function{callable.get()})) { return true; }
    }
    auto ops = kernel.propagated_builtin_callables();
    return ops.test(CallOp::MAKE_BFLOAT16_2) ||
           ops.test(CallOp::MAKE_BFLOAT16_3) ||
           ops.test(CallOp::MAKE_BFLOAT16_4);
}

}// namespace lc::vk::detail
