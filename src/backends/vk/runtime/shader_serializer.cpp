#include "shader_serializer.h"
#include "shader.h"
#include <backends/common/hlsl/shader_property.h>
#include "compute_shader.h"
#include <core/logging.h>

namespace lc::vk {
namespace detail {
struct ShaderSerHeader {
    vstd::MD5 md5;
    vstd::MD5 type_md5;
    uint3 block_size;
    uint64 property_size;
    uint64 spv_byte_size;
};
struct PSODataPackage {
    VkPipelineCacheHeaderVersionOne header;
    std::byte md5[sizeof(vstd::MD5)];
};
}// namespace detail
void ShaderSerializer::serialize_bytecode(
    Shader const *shader,
    vstd::MD5 shader_md5,
    vstd::MD5 type_md5,
    uint3 block_size,
    vstd::string_view file_name,
    vstd::span<const uint> spv_code,
    SerdeType serde_type,
    BinaryIO *bin_io) {
    using namespace detail;
    vstd::vector<std::byte> results;
    auto binds = shader->binds();
    ShaderSerHeader header{
        .md5 = shader_md5,
        .type_md5 = type_md5,
        .block_size = block_size,
        .property_size = binds.size(),
        .spv_byte_size = spv_code.size_bytes()};
    uint64_t final_size = sizeof(ShaderSerHeader) + header.property_size * sizeof(hlsl::Property) + header.spv_byte_size;
    results.push_back_uninitialized(final_size);
    auto data_ptr = results.data();
    auto save = [&]<typename T>(T const &t) {
        memcpy(data_ptr, &t, sizeof(T));
        data_ptr += sizeof(T);
    };
    auto save_arr = [&]<typename T>(T const *t, size_t size) {
        memcpy(data_ptr, t, sizeof(T) * size);
        data_ptr += sizeof(T) * size;
    };
    save(header);
    save_arr(binds.data(), binds.size());
    save_arr(spv_code.data(), spv_code.size());
    switch (serde_type) {
        case SerdeType::Cache:
            bin_io->write_shader_cache(file_name, results);
            break;
        case SerdeType::Builtin:
            bin_io->write_internal_shader(file_name, results);
            break;
        case SerdeType::ByteCode:
            bin_io->write_shader_bytecode(file_name, results);
            break;
    }
}
void ShaderSerializer::serialize_pso(
    Device *device,
    Shader const *shader,
    vstd::MD5 shader_md5,
    BinaryIO *bin_io) {
    vstd::vector<std::byte> pso_data;
    if (!shader->serialize_pso(pso_data)) return;
    using namespace detail;
    PSODataPackage package{
        .header = device->pso_header()};
    memcpy(package.md5, &shader_md5, sizeof(vstd::MD5));
    vstd::MD5 pso_md5{
        {reinterpret_cast<uint8_t const *>(&package), sizeof(PSODataPackage)}};
    auto file_name = pso_md5.to_string(false);
    bin_io->write_shader_cache(file_name, pso_data);
}
ComputeShader *ShaderSerializer::try_deser_compute(
    Device *device,
    // invalid md5 for AOT
    vstd::optional<vstd::MD5> shader_md5,
    vstd::vector<Argument> &&captured,
    vstd::MD5 type_md5,
    vstd::string_view file_name,
    SerdeType serde_type,
    BinaryIO *bin_io) {
    using namespace detail;
    vstd::vector<hlsl::Property> properties;
    vstd::vector<uint> spv;
    {
        auto read_stream = [&]() {
            switch (serde_type) {
                case SerdeType::Cache:
                    return bin_io->read_shader_cache(file_name);
                case SerdeType::Builtin:
                    return bin_io->read_internal_shader(file_name);
                case SerdeType::ByteCode:
                    return bin_io->read_shader_bytecode(file_name);
            }
        }();
        if (!read_stream) return nullptr;
        ShaderSerHeader header;
        if (read_stream->length() < sizeof(ShaderSerHeader)) return nullptr;
        read_stream->read({reinterpret_cast<std::byte *>(&header), sizeof(ShaderSerHeader)});
        if (read_stream->length() != (sizeof(ShaderSerHeader) + header.property_size * sizeof(hlsl::Property) + header.spv_byte_size))
            return nullptr;
        if (shader_md5 && *shader_md5 != header.md5)
            return nullptr;
        LUISA_ASSERT(header.type_md5 != type_md5, "Invalid shader type.");
        properties.push_back_uninitialized(header.property_size);
        read_stream->read({reinterpret_cast<std::byte *>(properties.data()), properties.size_bytes()});
        spv.push_back_uninitialized(header.spv_byte_size / sizeof(uint));
        read_stream->read({reinterpret_cast<std::byte *>(spv.data()), spv.size_bytes()});
    }
    vstd::vector<std::byte> pso_data;
    {
        PSODataPackage package{
            .header = device->pso_header()};
        memcpy(package.md5, &shader_md5, sizeof(vstd::MD5));
        vstd::MD5 pso_md5{
            {reinterpret_cast<uint8_t const *>(&package), sizeof(PSODataPackage)}};
        auto pso_name = pso_md5.to_string(false);
        auto read_stream = bin_io->read_shader_cache(pso_name);
        if (read_stream) {
            pso_data.push_back_uninitialized(read_stream->length());
            read_stream->read(pso_data);
        }
    }
    return new ComputeShader{
        device,
        properties,
        spv,
        std::move(captured),
        pso_data};
}
}// namespace lc::vk