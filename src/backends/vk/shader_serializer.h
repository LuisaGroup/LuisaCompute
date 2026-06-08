#pragma once
#include <luisa/vstl/md5.h>
#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include "../common/hlsl/shader_property.h"
#include "device.h"
#include "serde_type.h"
#include "shader.h"
namespace luisa {
class BinaryIO;
}// namespace luisa
namespace lc::vk {
using namespace luisa;
class RasterShader;
class Shader;
class ComputeShader;
class ShaderSerializer {
    ShaderSerializer() = delete;
    ~ShaderSerializer() = delete;
public:
    static void serialize_raster(
        vstd::span<const hlsl::Property> binds,
        vstd::span<const SavedArgument> saved_args,
        vstd::MD5 shader_md5,
        vstd::MD5 type_md5,
        vstd::string_view file_name,
        vstd::span<const uint> vert_spv_code,
        vstd::span<const uint> pixel_spv_code,
        SerdeType serde_type,
        BinaryIO const *bin_io,
        bool use_tex2d_bindless,
        bool use_tex3d_bindless,
        bool use_buffer_bindless,
        vstd::span<std::pair<vstd::string, luisa::compute::Type const *> const> printers,
        uint validation_count = 0);
    static void serialize_bytecode(
        vstd::span<const hlsl::Property> binds,
        vstd::span<const SavedArgument> saved_args,
        vstd::MD5 shader_md5,
        vstd::MD5 type_md5,
        uint3 block_size,
        vstd::string_view file_name,
        vstd::span<const uint> spv_code,
        SerdeType serde_type,
        BinaryIO const *bin_io,
        bool use_tex2d_bindless,
        bool use_tex3d_bindless,
        bool use_buffer_bindless,
        vstd::span<std::pair<vstd::string, luisa::compute::Type const *> const> printers,
        uint validation_count = 0);
    static void serialize_pso(
        Device *device,
        Shader const *shader,
        vstd::MD5 shader_md5,
        BinaryIO const *bin_io);

    struct DeserResult {
        Shader *shader;
        vstd::MD5 type_md5;
    };
    struct RasterDeserResult {
        RasterShader *shader;
        vstd::MD5 type_md5;
    };
    static DeserResult try_deser_compute(
        Device *device,
        // invalid md5 for AOT
        vstd::optional<vstd::MD5> shader_md5,
        vstd::vector<Argument> &&captured,
        vstd::string_view file_name,
        SerdeType serde_type,
        BinaryIO const *bin_io);
    static RasterDeserResult try_deser_raster(
        Device *device,
        // invalid md5 for AOT
        vstd::optional<vstd::MD5> shader_md5,
        vstd::vector<Argument> &&captured,
        vstd::string_view file_name,
        SerdeType serde_type,
        BinaryIO const *bin_io);
    static vstd::vector<SavedArgument> serialize_saved_args(Function kernel);
    static vstd::vector<SavedArgument> serialize_saved_args(vstd::IRange<std::pair<Variable, Usage>> &arguments);
};
}// namespace lc::vk
