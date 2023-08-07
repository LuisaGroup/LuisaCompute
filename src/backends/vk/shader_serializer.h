#pragma once
#include <luisa/vstl/md5.h>
#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include "../common/hlsl/shader_property.h"
#include "device.h"
#include "serde_type.h"
namespace luisa {
class BinaryIO;
}
namespace lc::vk {
using namespace luisa;
class Shader;
class ComputeShader;
class ShaderSerializer {
public:
    static void serialize_bytecode(
        vstd::span<const hlsl::Property> binds,
        vstd::MD5 shader_md5,
        vstd::MD5 type_md5,
        uint3 block_size,
        vstd::string_view file_name,
        vstd::span<const uint> spv_code,
        SerdeType serde_type,
        BinaryIO const *bin_io);
    static void serialize_pso(
        Device *device,
        Shader const *shader,
        vstd::MD5 shader_md5,
        BinaryIO const *bin_io);

    struct DeserResult {
        Shader *shader;
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
};
}// namespace lc::vk
