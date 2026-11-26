#pragma once

#include <cstdint>

#include <luisa/core/basic_types.h>
#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/osl/shader.h>
#include <luisa/osl/shader_group.h>
#include <luisa/osl/string_table.h>
#include <luisa/osl/translator_types.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>

namespace luisa::compute::osl {

class LUISA_OSL_API OSLTranslator {

public:
    struct Impl;

private:
    luisa::unique_ptr<Impl> _impl;

public:
    OSLTranslator() noexcept;
    virtual ~OSLTranslator() noexcept;
    OSLTranslator(const OSLTranslator &) noexcept = delete;
    OSLTranslator(OSLTranslator &&) noexcept = delete;
    OSLTranslator &operator=(const OSLTranslator &) noexcept = delete;
    OSLTranslator &operator=(OSLTranslator &&) noexcept = delete;

    [[nodiscard]] TranslationResult translate(
        const Shader &shader,
        xir::FunctionDefinition *target_function,
        const TranslatorOptions &options = {}) noexcept;

    [[nodiscard]] TranslationResult translate(
        const ShaderGroup &group,
        xir::FunctionDefinition *target_function,
        const TranslatorOptions &options = {}) noexcept;

protected:
    // Context & XIR helper utilities available inside subclass hooks
    [[nodiscard]] xir::Module *xir_module() const noexcept;
    [[nodiscard]] xir::FunctionDefinition *xir_function() const noexcept;
    [[nodiscard]] xir::XIRBuilder &xir_builder() noexcept;
    [[nodiscard]] const StringTable &string_table() const noexcept;
    void report_error(luisa::string message) noexcept;

    void store_named_option_output(const NamedOption &opt, const DualValue &value) noexcept;

    [[nodiscard]] xir::Value *const_int(int32_t v) noexcept;
    [[nodiscard]] xir::Value *const_uint(uint32_t v) noexcept;
    [[nodiscard]] xir::Value *const_float(float v) noexcept;
    [[nodiscard]] xir::Value *const_bool(bool v) noexcept;
    [[nodiscard]] xir::Value *const_float3(luisa::float3 v) noexcept;
    [[nodiscard]] xir::Value *const_float4(luisa::float4 v) noexcept;
    [[nodiscard]] xir::Value *const_float4x4(luisa::float4x4 v) noexcept;
    [[nodiscard]] xir::Value *zero_of(const compute::Type *type) noexcept;

    // 0. Lifecycle & closure XIR representation
    [[nodiscard]] virtual const compute::Type *closure_xir_type() const noexcept;
    virtual void on_begin_group(const ShaderGroup &group) noexcept {}
    virtual void on_end_group(const ShaderGroup &group,
                              luisa::span<const OutputBinding> outputs) noexcept {}

    // 1. Shading point global state
    [[nodiscard]] virtual DualValue emit_global_read(GlobalSymbol global, bool want_deriv) noexcept;
    virtual void emit_global_write(GlobalSymbol global, const DualValue &value) noexcept;
    virtual void emit_closure_output(const ClosureValue &ci) noexcept;
    [[nodiscard]] virtual xir::Value *emit_backfacing() noexcept;
    [[nodiscard]] virtual xir::Value *emit_raytype(luisa::string_view type_name) noexcept;

    // 2. Closure algebra (string name driven)
    [[nodiscard]] virtual ClosureValue emit_closure_zero() noexcept;
    [[nodiscard]] virtual ClosureValue emit_closure(
        luisa::string_view name,
        luisa::span<const HookArg> args,
        luisa::span<const NamedOption> opts) noexcept;
    [[nodiscard]] virtual ClosureValue emit_closure_scale(
        const ClosureValue &closure,
        xir::Value *color_weight) noexcept;
    [[nodiscard]] virtual ClosureValue emit_closure_add(
        const ClosureValue &lhs,
        const ClosureValue &rhs) noexcept;
    [[nodiscard]] virtual size_t closure_positional_arg_count(
        luisa::string_view name,
        luisa::span<const Type *const> arg_types) const noexcept;

    // 3. Attributes & messages (string name driven)
    [[nodiscard]] virtual xir::Value *emit_getattribute(
        luisa::optional<luisa::string_view> object_name,
        luisa::string_view attr_name,
        const Type *dst_type,
        int dst_array_len,
        bool want_deriv,
        DualValue &out_value) noexcept;
    [[nodiscard]] virtual xir::Value *emit_getmessage(
        luisa::optional<luisa::string_view> source,
        luisa::string_view msg_name,
        const Type *dst_type,
        int dst_array_len,
        bool want_deriv,
        DualValue &out_value) noexcept;
    virtual void emit_setmessage(
        luisa::string_view msg_name,
        const Type *src_type,
        int src_array_len,
        const DualValue &value) noexcept;

    // 4. Texture queries (filename + option string driven)
    [[nodiscard]] virtual DualValue emit_texture(
        TextureOp op,
        luisa::string_view filename,
        luisa::span<const DualValue> coords,
        luisa::span<const DualValue> explicit_derivs,
        const Type *dst_type,
        bool want_deriv,
        luisa::span<const NamedOption> opts) noexcept;
    [[nodiscard]] virtual xir::Value *emit_gettextureinfo(
        luisa::string_view filename,
        luisa::string_view dataname,
        const Type *dst_type,
        int dst_array_len,
        DualValue &out_value) noexcept;

    // 5. Coordinate space transforms (space name string driven)
    [[nodiscard]] virtual xir::Value *emit_getmatrix(
        luisa::string_view from_space,
        luisa::string_view to_space,
        xir::Value *&out_matrix) noexcept;
    [[nodiscard]] virtual DualValue emit_transform_space(
        luisa::string_view from_space,
        luisa::string_view to_space,
        TransformKind kind,
        const DualValue &value,
        bool want_deriv) noexcept;

    // 6. Color space transforms (default supports rgb/linear/Rec709/sRGB/hsv/hsl/XYZ/xyY/YIQ)
    [[nodiscard]] virtual DualValue emit_transform_color(
        luisa::string_view from_space,
        luisa::string_view to_space,
        const DualValue &value,
        bool want_deriv) noexcept;

    // 7. Noise (default supports perlin/uperlin/simplex/usimplex/cell/hash + periodic)
    [[nodiscard]] virtual DualValue emit_noise(
        luisa::string_view noise_type,
        NoiseOp op,
        luisa::span<const DualValue> coords,
        luisa::span<const DualValue> periods,
        const Type *dst_type,
        bool want_deriv,
        luisa::span<const NamedOption> opts) noexcept;

    // 8. Ray queries & string interning
    [[nodiscard]] virtual xir::Value *emit_trace(
        const DualValue &pos,
        const DualValue &dir,
        luisa::span<const NamedOption> opts) noexcept;
    [[nodiscard]] virtual uint32_t intern_string(luisa::string_view s) noexcept;
};

}// namespace luisa::compute::osl
