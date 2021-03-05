#pragma once

#include <span>
#include <variant>

#include <core/data_types.h>
#include <ast/variable.h>

namespace luisa::compute {

class FunctionBuilder;
class ScopeStmt;

class Function {

public:
    enum struct Tag {
        KERNEL,
        CALLABLE,
        // TODO: Ray-tracing functions...
    };

    struct BufferBinding {
        Variable variable;
        uint64_t handle;
        size_t offset_bytes;
    };

    struct TextureBinding {
        // TODO...
    };
    
    struct ConstantData {
        using Ptr = std::variant<
            bool const *, float const *, char const *, uchar const *, short const *, ushort const *, int const *, uint const *,
            bool2 const *, float2 const *, char2 const *, uchar2 const *, short2 const *, ushort2 const *, int2 const *, uint2 const *,
            bool3 const *, float3 const *, char3 const *, uchar3 const *, short3 const *, ushort3 const *, int3 const *, uint3 const *,
            bool4 const *, float4 const *, char4 const *, uchar4 const *, short4 const *, ushort4 const *, int4 const *, uint4 const *,
            float3x3 const *, float4x4 const *>;
        Variable variable;
        Ptr data;
    };

private:
    const FunctionBuilder &_builder;

public:
    Function(const FunctionBuilder &builder) noexcept : _builder{builder} {}
    [[nodiscard]] std::span<const Variable> builtin_variables() const noexcept;
    [[nodiscard]] std::span<const Variable> shared_variables() const noexcept;
    [[nodiscard]] std::span<const ConstantData> constant_variables() const noexcept;
    [[nodiscard]] std::span<const BufferBinding> captured_buffers() const noexcept;
    [[nodiscard]] std::span<const TextureBinding> captured_textures() const noexcept;
    [[nodiscard]] std::span<const Variable> arguments() const noexcept;
    [[nodiscard]] std::span<const uint32_t> custom_callables() const noexcept;
    [[nodiscard]] std::span<const std::string_view> builtin_callables() const noexcept;
    [[nodiscard]] Tag tag() const noexcept;
    [[nodiscard]] uint32_t uid() const noexcept;
    [[nodiscard]] const Type *return_type() const noexcept;
    [[nodiscard]] const ScopeStmt *body() const noexcept;
    [[nodiscard]] static Function callable(uint32_t uid) noexcept;
    [[nodiscard]] static Function kernel(uint32_t uid) noexcept;
};

}// namespace luisa::compute
