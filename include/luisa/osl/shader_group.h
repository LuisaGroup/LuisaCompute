#pragma once

#include <array>
#include <cstdint>

#include <luisa/core/basic_types.h>
#include <luisa/core/dll_export.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/variant.h>
#include <luisa/core/stl/vector.h>
#include <luisa/osl/shader.h>

namespace luisa::compute::osl {

struct LUISA_OSL_API ParameterValue {
    using Storage = luisa::variant<
        int32_t,
        float,
        luisa::float3,
        luisa::float4x4,
        luisa::string,
        luisa::vector<int32_t>,
        luisa::vector<float>,
        luisa::vector<luisa::float3>,
        luisa::vector<luisa::float4x4>,
        luisa::vector<luisa::string>>;

    Storage value;

    ParameterValue() noexcept : value{int32_t{0}} {}
    ParameterValue(int32_t v) noexcept : value{v} {}
    ParameterValue(float v) noexcept : value{v} {}
    ParameterValue(luisa::float3 v) noexcept : value{v} {}
    ParameterValue(luisa::float4x4 v) noexcept : value{v} {}
    ParameterValue(luisa::string v) noexcept : value{std::move(v)} {}
    ParameterValue(luisa::string_view v) noexcept : value{luisa::string{v}} {}
    ParameterValue(const char *v) noexcept : value{luisa::string{v ? v : ""}} {}
    ParameterValue(luisa::vector<int32_t> v) noexcept : value{std::move(v)} {}
    ParameterValue(luisa::vector<float> v) noexcept : value{std::move(v)} {}
    ParameterValue(luisa::vector<luisa::float3> v) noexcept : value{std::move(v)} {}
    ParameterValue(luisa::vector<luisa::float4x4> v) noexcept : value{std::move(v)} {}
    ParameterValue(luisa::vector<luisa::string> v) noexcept : value{std::move(v)} {}

    // Converts an OSL 4x4 row-major matrix (16 floats, row 3 = translation)
    // to LuisaCompute's column-major float4x4 (L = M^T).
    [[nodiscard]] static ParameterValue from_osl_row_major(const std::array<float, 16> &m) noexcept;
};

struct ShaderConnection {
    uint32_t src_layer{0u};
    luisa::string src_param;
    uint32_t dst_layer{0u};
    luisa::string dst_param;
};

struct ShaderOutputRequest {
    uint32_t layer{0u};
    luisa::string param;
    bool want_deriv{false};
};

struct ShaderLayer {
    luisa::string name;
    const Shader *shader{nullptr};
    luisa::unordered_map<luisa::string, ParameterValue> overrides;
    luisa::unordered_map<luisa::string, bool> dynamic_inputs;
};

class LUISA_OSL_API ShaderGroup {

private:
    luisa::vector<ShaderLayer> _layers;
    luisa::vector<ShaderConnection> _connections;
    luisa::vector<ShaderOutputRequest> _output_requests;
    luisa::optional<uint32_t> _root_layer;

public:
    ShaderGroup() noexcept = default;
    ~ShaderGroup() noexcept = default;
    ShaderGroup(const ShaderGroup &) noexcept = default;
    ShaderGroup(ShaderGroup &&) noexcept = default;
    ShaderGroup &operator=(const ShaderGroup &) noexcept = default;
    ShaderGroup &operator=(ShaderGroup &&) noexcept = default;

    uint32_t add_layer(luisa::string name, const Shader *shader) noexcept;
    uint32_t add_layer(const Shader *shader) noexcept;
    void set_parameter(uint32_t layer, luisa::string_view param_name, ParameterValue value) noexcept;
    void mark_dynamic_input(uint32_t layer, luisa::string_view param_name, bool has_deriv = false) noexcept;
    void connect_shaders(uint32_t src_layer, luisa::string_view src_param,
                         uint32_t dst_layer, luisa::string_view dst_param) noexcept;
    void mark_output(uint32_t layer, luisa::string_view param_name, bool want_deriv = false) noexcept;
    void set_root_layer(uint32_t layer) noexcept;
    void clear_root_layer() noexcept { _root_layer.reset(); }

    [[nodiscard]] luisa::span<const ShaderLayer> layers() const noexcept { return _layers; }
    [[nodiscard]] luisa::span<const ShaderConnection> connections() const noexcept { return _connections; }
    [[nodiscard]] luisa::span<const ShaderOutputRequest> output_requests() const noexcept { return _output_requests; }
    [[nodiscard]] luisa::optional<uint32_t> root_layer() const noexcept { return _root_layer; }
    [[nodiscard]] luisa::optional<uint32_t> effective_root_layer() const noexcept;
};

}// namespace luisa::compute::osl
