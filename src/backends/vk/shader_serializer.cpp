#include "shader_serializer.h"
#include "shader.h"
#include "../common/hlsl/shader_property.h"
#include "../common/hlsl/hlsl_codegen.h"
#include "compute_shader.h"
#include "raster_shader.h"

namespace lc::vk {
namespace detail {
struct ShaderSerHeader {
    uint64 header_ver;
    vstd::MD5 md5;
    vstd::MD5 type_md5;
    uint64 property_size;
    uint64 spv_byte_size;
    uint block_size[3];
    uint kernel_arg_count;
    uint printer_count;
    uint printer_size_bytes;
    uint validation_count;
    bool use_bindless_buffer;
    bool use_bindless_tex2d;
    bool use_bindless_tex3d;
};
struct RasterSerHeader {
    uint64 header_ver;
    vstd::MD5 md5;
    vstd::MD5 type_md5;
    uint64 property_size;
    uint64 vert_spv_byte_size;
    uint64 pixel_spv_byte_size;
    uint kernel_arg_count;
    uint printer_count;
    uint printer_size_bytes;
    uint validation_count;
    bool use_bindless_buffer;
    bool use_bindless_tex2d;
    bool use_bindless_tex3d;
};
constexpr uint32_t kShaderSerVersion = 2; // version: set to header_ver
struct PSODataPackage {
    VkPipelineCacheHeaderVersionOne header;
    std::byte md5[sizeof(vstd::MD5)];
};
}// namespace detail
class StringViewBinaryStream : public BinaryStream {

public:
    luisa::string_view strv;
    explicit StringViewBinaryStream(luisa::string_view strv) : strv(strv) {}
    [[nodiscard]] size_t length() const noexcept override { return strv.size(); }
    [[nodiscard]] size_t pos() const noexcept override { return _pos; }
    void read(luisa::span<std::byte> dst) noexcept override {
        LUISA_DEBUG_ASSERT(dst.size() + _pos <= strv.size());
        std::memcpy(dst.data(), strv.data() + _pos, dst.size());
        _pos += dst.size();
    }
    ~StringViewBinaryStream() noexcept override = default;

private:
    size_t _pos{};
};

luisa::unique_ptr<luisa::BinaryStream> read_binary_io(SerdeType type, luisa::BinaryIO const *bin_io, luisa::string_view file_name) {
    switch (type) {
        case SerdeType::kCache:
            return bin_io->read_shader_cache(file_name);
        case SerdeType::kBuiltin: {
            auto internal_data = hlsl::CodegenUtility::ReadInternalHLSLFile(file_name);
            if (!internal_data.empty()) {
                return luisa::make_unique<StringViewBinaryStream>(internal_data);
            }
            return bin_io->read_internal_shader(file_name);
        }
        case SerdeType::kByteCode:
            return bin_io->read_shader_bytecode(file_name);
    }
    return luisa::unique_ptr<luisa::BinaryStream>{};
}
void ShaderSerializer::serialize_raster(
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
    vstd::span<std::pair<vstd::string, Type const *> const> printers,
    uint validation_count) {
    vstd::vector<std::byte> results;
    using namespace detail;
    RasterSerHeader header{
        .header_ver = kShaderSerVersion,
        .md5 = shader_md5,
        .type_md5 = type_md5,
        .property_size = binds.size(),
        .vert_spv_byte_size = vert_spv_code.size_bytes(),
        .pixel_spv_byte_size = pixel_spv_code.size_bytes(),
        .kernel_arg_count = (uint)saved_args.size()};
    uint printer_size_bytes = 0;
    for (auto &i : printers) {
        printer_size_bytes += i.first.size() + i.second->description().size() + 2;// zero-end string
    }
    uint64_t final_size = sizeof(RasterSerHeader) +
                          header.property_size * sizeof(hlsl::Property) +
                          header.vert_spv_byte_size +
                          header.pixel_spv_byte_size +
                          header.kernel_arg_count * sizeof(SavedArgument) +
                          printer_size_bytes;
    luisa::enlarge_by(results, final_size);
    auto data_ptr = results.data();
    auto save = [&]<typename T>(T const &t) {
        memcpy(data_ptr, &t, sizeof(T));
        data_ptr += sizeof(T);
    };
    auto save_arr = [&]<typename T>(T const *t, size_t size) {
        memcpy(data_ptr, t, sizeof(T) * size);
        data_ptr += sizeof(T) * size;
    };
    header.use_bindless_buffer = use_buffer_bindless;
    header.use_bindless_tex2d = use_tex2d_bindless;
    header.use_bindless_tex3d = use_tex3d_bindless;
    header.printer_count = printers.size();
    header.printer_size_bytes = printer_size_bytes;
    header.validation_count = validation_count;
    save(header);
    save_arr(binds.data(), binds.size());
    save_arr(saved_args.data(), saved_args.size());
    save_arr(vert_spv_code.data(), vert_spv_code.size());
    save_arr(pixel_spv_code.data(), pixel_spv_code.size());
    for (auto &i : printers) {
        save_arr(i.first.c_str(), i.first.size() + 1);
        auto desc = i.second->description();
        save_arr(desc.data(), desc.size());
        *data_ptr = (std::byte)0;
        ++data_ptr;
    }

    switch (serde_type) {
        case SerdeType::kCache:
            static_cast<void>(bin_io->write_shader_cache(file_name, results));
            break;
        case SerdeType::kBuiltin:
            static_cast<void>(bin_io->write_internal_shader(file_name, results));
            break;
        case SerdeType::kByteCode:
            static_cast<void>(bin_io->write_shader_bytecode(file_name, results));
            break;
    }
}
void ShaderSerializer::serialize_bytecode(
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
    vstd::span<std::pair<vstd::string, Type const *> const> printers,
    uint validation_count) {
    using namespace detail;
    vstd::vector<std::byte> results;
    ShaderSerHeader header{
        .header_ver = kShaderSerVersion,
        .md5 = shader_md5,
        .type_md5 = type_md5,
        .property_size = binds.size(),
        .spv_byte_size = spv_code.size_bytes(),
        .block_size = {block_size.x, block_size.y, block_size.z},
        .kernel_arg_count = (uint)saved_args.size()};
    uint printer_size_bytes = 0;
    for (auto &i : printers) {
        printer_size_bytes += i.first.size() + i.second->description().size() + 2;// zero-end string
    }
    uint64_t final_size = sizeof(ShaderSerHeader) +
                          header.property_size * sizeof(hlsl::Property) +
                          header.spv_byte_size +
                          header.kernel_arg_count * sizeof(SavedArgument) +
                          printer_size_bytes;
    luisa::enlarge_by(results, final_size);
    auto data_ptr = results.data();
    auto save = [&]<typename T>(T const &t) {
        memcpy(data_ptr, &t, sizeof(T));
        data_ptr += sizeof(T);
    };
    auto save_arr = [&]<typename T>(T const *t, size_t size) {
        memcpy(data_ptr, t, sizeof(T) * size);
        data_ptr += sizeof(T) * size;
    };
    header.use_bindless_buffer = use_buffer_bindless;
    header.use_bindless_tex2d = use_tex2d_bindless;
    header.use_bindless_tex3d = use_tex3d_bindless;
    header.printer_count = printers.size();
    header.printer_size_bytes = printer_size_bytes;
    header.validation_count = validation_count;
    save(header);
    save_arr(binds.data(), binds.size());
    save_arr(saved_args.data(), saved_args.size());
    save_arr(spv_code.data(), spv_code.size());
    for (auto &i : printers) {
        save_arr(i.first.c_str(), i.first.size() + 1);
        auto desc = i.second->description();
        save_arr(desc.data(), desc.size());
        *data_ptr = (std::byte)0;
        ++data_ptr;
    }

    switch (serde_type) {
        case SerdeType::kCache:
            bin_io->write_shader_cache(file_name, results);
            break;
        case SerdeType::kBuiltin:
            bin_io->write_internal_shader(file_name, results);
            break;
        case SerdeType::kByteCode:
            bin_io->write_shader_bytecode(file_name, results);
            break;
    }
}
void ShaderSerializer::serialize_pso(
    Device *device,
    Shader const *shader,
    vstd::MD5 shader_md5,
    BinaryIO const *bin_io) {
    vstd::vector<std::byte> pso_data;
    if (!shader->serialize_pso(pso_data)) return;
    using namespace detail;
    PSODataPackage package{
        .header = device->pso_header()};
    memcpy(package.md5, &shader_md5, sizeof(vstd::MD5));
    vstd::MD5 pso_md5{
        {reinterpret_cast<uint8_t const *>(&package), sizeof(PSODataPackage)}};
    auto file_name = pso_md5.to_string(false) + ".vk";

    bin_io->write_shader_cache(file_name, pso_data);
}
auto ShaderSerializer::try_deser_raster(
    Device *device,
    // invalid md5 for AOT
    vstd::optional<vstd::MD5> shader_md5,
    vstd::vector<Argument> &&captured,
    vstd::string_view file_name,
    SerdeType serde_type,
    BinaryIO const *bin_io) -> RasterDeserResult {
    vstd::vector<hlsl::Property> properties;
    vstd::vector<SavedArgument> saved_args;
    vstd::vector<uint> vert_spv;
    vstd::vector<uint> pixel_spv;
    RasterDeserResult result{
        .shader = nullptr};
    [[maybe_unused]] uint3 block_size;
    using namespace detail;
    RasterSerHeader header;
    vstd::vector<std::pair<luisa::string, Type const *>> printers;
    {
        auto read_stream = [&]() {
            return read_binary_io(serde_type, bin_io, file_name);
        }();
        if (!read_stream) return result;
        auto stream_len = read_stream->length();
        if (stream_len < sizeof(RasterSerHeader)) return result;
        read_stream->read({reinterpret_cast<std::byte *>(&header), sizeof(RasterSerHeader)});
        if (header.header_ver != kShaderSerVersion) return result;
        uint64_t final_size = sizeof(RasterSerHeader) +
                              header.property_size * sizeof(hlsl::Property) +
                              header.vert_spv_byte_size +
                              header.pixel_spv_byte_size +
                              header.kernel_arg_count * sizeof(SavedArgument) +
                              header.printer_size_bytes;
        if (final_size != stream_len) return result;
        if (shader_md5 && *shader_md5 != header.md5)
            return result;
        result.type_md5 = header.type_md5;
        luisa::enlarge_by(properties, header.property_size);
        read_stream->read({reinterpret_cast<std::byte *>(properties.data()), luisa::size_bytes(properties)});
        luisa::enlarge_by(saved_args, header.kernel_arg_count);
        read_stream->read({reinterpret_cast<std::byte *>(saved_args.data()), luisa::size_bytes(saved_args)});
        luisa::enlarge_by(vert_spv, header.vert_spv_byte_size / sizeof(uint));
        read_stream->read({reinterpret_cast<std::byte *>(vert_spv.data()), luisa::size_bytes(vert_spv)});
        luisa::enlarge_by(pixel_spv, header.pixel_spv_byte_size / sizeof(uint));
        read_stream->read({reinterpret_cast<std::byte *>(pixel_spv.data()), luisa::size_bytes(pixel_spv)});
        luisa::vector<char> printer_data;
        printers.reserve(header.printer_count);
        if (header.printer_size_bytes > 0) {
            luisa::enlarge_by(printer_data, header.printer_size_bytes);
            read_stream->read({reinterpret_cast<std::byte *>(printer_data.data()),
                               printer_data.size()});
            auto ptr = printer_data.data();
            for (auto _ : vstd::range(header.printer_count)) {
                auto printer_len = strlen(ptr);
                luisa::string_view printer_val{ptr, printer_len};
                ptr += printer_len + 1;
                auto type_desc_len = strlen(ptr);
                auto type = Type::from(luisa::string_view{ptr, type_desc_len});
                ptr += type_desc_len + 1;
                printers.emplace_back(luisa::string{printer_val}, type);
            }
        }
    }
    result.shader = new RasterShader(
        device,
        std::move(captured),
        std::move(saved_args),
        properties,
        {},
        std::move(vert_spv),
        std::move(pixel_spv),
        header.use_bindless_tex2d,
        header.use_bindless_tex3d,
        header.use_bindless_buffer,
        header.validation_count);
    return result;
}

ShaderSerializer::DeserResult ShaderSerializer::try_deser_compute(
    Device *device,
    // invalid md5 for AOT
    vstd::optional<vstd::MD5> shader_md5,
    vstd::vector<Argument> &&captured,
    vstd::string_view file_name,
    SerdeType serde_type,
    BinaryIO const *bin_io) {
    using namespace detail;
    vstd::vector<hlsl::Property> properties;
    vstd::vector<SavedArgument> saved_args;
    vstd::vector<uint> spv;
    DeserResult result{
        .shader = nullptr};
    uint3 block_size;
    ShaderSerHeader header;
    vstd::vector<std::pair<luisa::string, Type const *>> printers;
    {
        auto read_stream = [&]() {
            return read_binary_io(serde_type, bin_io, file_name);
        }();
        if (!read_stream) return result;
        auto stream_len = read_stream->length();
        if (stream_len < sizeof(ShaderSerHeader)) return result;
        read_stream->read({reinterpret_cast<std::byte *>(&header), sizeof(ShaderSerHeader)});
        if (header.header_ver != kShaderSerVersion) return result;
        if (stream_len != read_stream->pos() + (header.printer_size_bytes + header.property_size * sizeof(hlsl::Property) + header.spv_byte_size + header.kernel_arg_count * sizeof(SavedArgument)))
            return result;
        if (shader_md5 && *shader_md5 != header.md5)
            return result;
        result.type_md5 = header.type_md5;
        block_size = uint3(header.block_size[0], header.block_size[1], header.block_size[2]);
        luisa::enlarge_by(properties, header.property_size);
        read_stream->read({reinterpret_cast<std::byte *>(properties.data()), luisa::size_bytes(properties)});
        luisa::enlarge_by(saved_args, header.kernel_arg_count);
        read_stream->read({reinterpret_cast<std::byte *>(saved_args.data()), luisa::size_bytes(saved_args)});
        luisa::enlarge_by(spv, header.spv_byte_size / sizeof(uint));
        read_stream->read({reinterpret_cast<std::byte *>(spv.data()), luisa::size_bytes(spv)});
        luisa::vector<char> printer_data;
        printers.reserve(header.printer_count);
        if (header.printer_size_bytes > 0) {
            luisa::enlarge_by(printer_data, header.printer_size_bytes);
            read_stream->read({reinterpret_cast<std::byte *>(printer_data.data()),
                               printer_data.size()});
            auto ptr = printer_data.data();
            for (auto _ : vstd::range(header.printer_count)) {
                auto printer_len = strlen(ptr);
                luisa::string_view printer_val{ptr, printer_len};
                ptr += printer_len + 1;
                auto type_desc_len = strlen(ptr);
                auto type = Type::from(luisa::string_view{ptr, type_desc_len});
                ptr += type_desc_len + 1;
                printers.emplace_back(luisa::string{printer_val}, type);
            }
        }
    }
    vstd::vector<std::byte> pso_data;
    vstd::string pso_name;
    {
        PSODataPackage package{
            .header = device->pso_header()};
        if (!shader_md5) {
            shader_md5.create(vstd::MD5{file_name});
        }
        std::memcpy(package.md5, shader_md5.ptr(), sizeof(vstd::MD5));
        vstd::MD5 pso_md5{
            {reinterpret_cast<uint8_t const *>(&package), sizeof(PSODataPackage)}};
        pso_name = pso_md5.to_string(false) + ".vk";
        auto read_stream = bin_io->read_shader_cache(pso_name);
        if (read_stream) {
            auto stream_len = read_stream->length();
            if (stream_len >= sizeof(VkPipelineCacheHeaderVersionOne)) {
                luisa::enlarge_by(pso_data, sizeof(VkPipelineCacheHeaderVersionOne));
                read_stream->read(pso_data);
                if (!device->is_pso_same(*reinterpret_cast<VkPipelineCacheHeaderVersionOne const *>(pso_data.data()))) {
                    pso_data.clear();
                } else {
                    auto last_size = stream_len - read_stream->pos();
                    luisa::enlarge_by(pso_data, last_size);
                    read_stream->read({pso_data.data() + sizeof(VkPipelineCacheHeaderVersionOne), last_size});
                }
            }
        }
    }
    auto shader = new ComputeShader{
        device,
        block_size,
        properties,
        std::move(saved_args),
        spv,
        std::move(captured),
        pso_data,
        header.use_bindless_tex2d,
        header.use_bindless_tex3d,
        header.use_bindless_buffer,
        std::move(printers),
        {},
        header.validation_count};
    if (pso_data.empty() &&
        shader->serialize_pso(pso_data)) {
        bin_io->write_shader_cache(pso_name, pso_data);
    }
    result.shader = shader;
    return result;
}
vstd::vector<SavedArgument> ShaderSerializer::serialize_saved_args(Function kernel) {
    LUISA_ASSUME(kernel.tag() != Function::Tag::CALLABLE);
    auto &&args = kernel.arguments();
    vstd::vector<SavedArgument> result;
    vstd::push_back_func(result, args.size(), [&](size_t i) {
        auto &&var = args[i];
        return SavedArgument(kernel, var);
    });
    return result;
}

vstd::vector<SavedArgument> ShaderSerializer::serialize_saved_args(
    vstd::IRange<std::pair<Variable, Usage>> &arguments) {
    vstd::vector<SavedArgument> result;
    for (auto &&i : arguments) {
        result.emplace_back(i.second, i.first);
    }
    return result;
}

}// namespace lc::vk
