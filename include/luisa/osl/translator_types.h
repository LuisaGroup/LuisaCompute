#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/basic_types.h>
#include <luisa/core/dll_export.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/variant.h>
#include <luisa/core/stl/vector.h>
#include <luisa/osl/string_table.h>

namespace luisa::compute::xir {
class Value;
}// namespace luisa::compute::xir

namespace luisa::compute::osl {

class Type;
class Symbol;

struct DualValue {
    xir::Value *val{nullptr};
    xir::Value *dx{nullptr};// nullptr represents static zero derivative
    xir::Value *dy{nullptr};

    [[nodiscard]] bool has_derivs() const noexcept { return dx != nullptr || dy != nullptr; }
    [[nodiscard]] bool valid() const noexcept { return val != nullptr; }
};

struct ClosureValue {
    xir::Value *value{nullptr};

    [[nodiscard]] bool valid() const noexcept { return value != nullptr; }
};

using ConstantOptionValue = luisa::variant<
    int32_t,
    float,
    luisa::float3,
    luisa::float4x4,
    luisa::string_view>;

struct HookArg {
    const Type *osl_type{nullptr};
    int array_length{0};
    DualValue value{};
    ClosureValue closure{};
    luisa::optional<ConstantOptionValue> constant{};
};

struct NamedOption {
    luisa::string_view key;
    const Type *osl_type{nullptr};
    int array_length{0};
    bool is_output{false};
    DualValue value{};
    xir::Value *out_val_ptr{nullptr};
    xir::Value *out_dx_ptr{nullptr};
    xir::Value *out_dy_ptr{nullptr};
    luisa::optional<ConstantOptionValue> constant{};
};

enum struct GlobalSymbol : uint32_t {
    P,
    I,
    N,
    Ng,
    u,
    v,
    dPdu,
    dPdv,
    Ps,
    time,
    dtime,
    dPdtime,
    Ci,
};

enum struct TransformKind : uint32_t {
    POINT,
    VECTOR,
    NORMAL,
    MATRIX,
};

enum struct TextureOp : uint32_t {
    TEXTURE_2D,
    TEXTURE_3D,
    ENVIRONMENT,
};

enum struct NoiseOp : uint32_t {
    NOISE,
    PNOISE,
    SNOISE,
    PSNOISE,
    CELLNOISE,
    HASHNOISE,
};

enum struct BoundsCheckMode : uint32_t {
    CLAMP,
    GUARD,
};

struct OutputBinding {
    uint32_t layer_index{0u};
    luisa::string_view layer_name;
    const Symbol *symbol{nullptr};
    luisa::string_view field_name;
    const Type *type{nullptr};
    int array_length{0};
    DualValue value{};
    ClosureValue closure{};
    luisa::vector<OutputBinding> elements{};
    luisa::vector<OutputBinding> fields{};
};

struct TranslatorOptions {
    uint32_t max_string_lattice_size{32u};
    uint32_t max_string_switch_cases{32u};
    bool verify_xir{true};
    bool run_xir_passes{true};
    bool restructure_for_ast{false};
    bool prune_dead_layers{true};
    BoundsCheckMode bounds_check{BoundsCheckMode::CLAMP};
};

struct TranslationError {
    luisa::string layer_name;
    uint32_t instruction_index{~0u};
    luisa::string opcode;
    luisa::string filename;
    int line{-1};
    luisa::string message;
};

struct TranslationResult {
    luisa::vector<TranslationError> errors;
    StringTable string_table;
    size_t active_layer_count{0u};
    size_t eliminated_layer_count{0u};
    size_t dual_symbol_count{0u};
    size_t string_switch_count{0u};

    [[nodiscard]] bool succeeded() const noexcept { return errors.empty(); }
};

}// namespace luisa::compute::osl
