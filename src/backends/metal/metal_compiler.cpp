//
// Created by Mike Smith on 2023/4/15.
//

#include <fstream>
#include <backends/metal/metal_compiler.h>

namespace luisa::compute::metal {

namespace detail {

[[nodiscard]] static auto temp_unique_file_path() noexcept {
    std::error_code ec;
    auto temp_dir = std::filesystem::temp_directory_path(ec);
    std::filesystem::path temp_path;
    if (!ec) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to find temporary directory: {}.",
            ec.message());
    } else {
        auto uuid = CFUUIDCreate(nullptr);
        auto uuid_string = CFUUIDCreateString(nullptr, uuid);
        temp_path = temp_dir / CFStringGetCStringPtr(uuid_string, kCFStringEncodingUTF8);
        CFRelease(uuid);
        CFRelease(uuid_string);
    }
    return temp_path;
}

}// namespace detail

MetalCompiler::MetalCompiler(const MetalDevice *device) noexcept
    : _device{device}, _cache{max_cache_item_count} {}

struct PrecompiledShaderHeader {
    uint64_t hash;
    uint3 block_size;
};

void MetalCompiler::_store_disk_archive(uint64_t hash, luisa::string_view name,
                                        const ShaderOption &option, uint3 block_size,
                                        MTL::ComputePipelineDescriptor *pipeline_desc) const noexcept {

    // create a binary archive
    NS::Error *error = nullptr;
    auto archive_desc = MTL::BinaryArchiveDescriptor::alloc()->init();
    auto archive = NS::TransferPtr(_device->handle()->newBinaryArchive(archive_desc, &error));
    archive_desc->release();
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal shader "
            "archive for '{}': {}.",
            name, error->localizedDescription()->utf8String());
        return;
    }
    archive->addComputePipelineFunctions(pipeline_desc, &error);
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal shader "
            "archive for '{}': {}.",
            name, error->localizedDescription()->utf8String());
        return;
    }

    // dump library
    auto temp_file_path = detail::temp_unique_file_path();
    if (temp_file_path.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal shader "
            "archive for '{}': failed to create temporary file.",
            name);
        return;
    }
    auto url = NS::URL::fileURLWithPath(NS::String::string(
        temp_file_path.c_str(), NS::UTF8StringEncoding));
    archive->serializeToURL(url, &error);
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal shader "
            "archive for '{}': {}.",
            name, error->localizedDescription()->utf8String());
        return;
    }

    // read the dumped library
    std::error_code ec;
    auto file_size = std::filesystem::file_size(temp_file_path, ec);
    if (ec) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal shader "
            "archive for '{}': {}.",
            name, ec.message());
        return;
    }
    PrecompiledShaderHeader header{
        .hash = hash,
        .block_size = block_size};
    luisa::vector<std::byte> buffer;
    buffer.resize(sizeof(header) + file_size);
    std::memcpy(buffer.data(), &header, sizeof(header));
    std::ifstream file{temp_file_path, std::ios::binary};
    if (!file.is_open()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal shader "
            "archive for '{}': failed to open temporary file.",
            name);
        return;
    }
    file.read(reinterpret_cast<char *>(buffer.data() + sizeof(header)),
              static_cast<ssize_t>(file_size));
    file.close();

    // store the binary archive
    auto io = _device->io();
    if (!option.name.empty()) {
        io->write_shader_bytecode(name, buffer);
    } else {
        io->write_shader_cache(name, buffer);
    }
}

std::pair<MTL::Library *, uint3 /* block size */>
MetalCompiler::_load_disk_archive(uint64_t hash,
                                  luisa::string_view name,
                                  const ShaderOption &option,
                                  uint3 block_size) const noexcept {

    // open file stream
    auto io = _device->io();
    auto stream = !option.name.empty() ? io->read_shader_bytecode(name) :
                  option.enable_cache  ? io->read_shader_cache(name) :
                                         nullptr;
    if (stream == nullptr || stream->length() <= sizeof(PrecompiledShaderHeader)) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal shader "
            "archive for '{}': file not found.",
            name);
        return std::make_pair(nullptr, make_uint3());
    }

    // load data
    luisa::vector<std::byte> buffer(stream->length());
    stream->read(buffer);
    stream.reset();

    // check hash
    PrecompiledShaderHeader header{};
    std::memcpy(&header, buffer.data(), sizeof(header));
    if (header.hash != hash) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal shader "
            "archive for '{}': hash mismatch.",
            name);
        return std::make_pair(nullptr, make_uint3());
    }
    if (!all(block_size == 0u) && !all(block_size == header.block_size)) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal shader "
            "archive for '{}': block size mismatch.",
            name);
        return std::make_pair(nullptr, make_uint3());
    }

    // load library
    auto library_data = luisa::span{buffer}.subspan(sizeof(uint64_t));
    auto dispatch_data = dispatch_data_create(library_data.data(), library_data.size(), nullptr, nullptr);
    NS::Error *error = nullptr;
    auto library = _device->handle()->newLibrary(dispatch_data, &error);
    dispatch_release(dispatch_data);
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal shader "
            "archive for '{}': {}.",
            name, error->localizedDescription()->utf8String());
        return std::make_pair(nullptr, make_uint3());
    }
    return std::make_pair(library, header.block_size);
}

NS::SharedPtr<MTL::ComputePipelineState> MetalCompiler::compile(
    luisa::string_view src, const ShaderOption &option, uint3 block_size) const noexcept {

    return with_autorelease_pool([&] {
        auto src_hash = luisa::hash_value(src);
        auto opt_hash = luisa::hash_value(option);
        auto hash = luisa::hash_combine({src_hash, opt_hash});

        // try memory cache
        if (auto pso = _cache.fetch(hash)) { return *pso; }

        // name
        auto name = option.name.empty() ?
                        luisa::format("metal_kernel_{:016x}", hash) :
                        option.name;

        auto uses_cache = !option.name.empty() || option.enable_cache;

        // try disk cache
        auto library = uses_cache ?
                           _load_disk_archive(hash, name, option, block_size).first :
                           nullptr;

        if (library == nullptr) {

            if (uses_cache) {
                LUISA_INFO("Failed to load Metal shader archive for '{}'. "
                           "Falling back to compilation from source.",
                           name);
            }
            auto source = NS::String::alloc()->init(const_cast<char *>(src.data()),
                                                    src.size(), NS::UTF8StringEncoding, false);
            auto options = MTL::CompileOptions::alloc()->init();
            options->setFastMathEnabled(option.enable_fast_math);
            options->setLanguageVersion(MTL::LanguageVersion3_0);
            options->setLibraryType(MTL::LibraryTypeExecutable);
            NS::Error *error;
            library = _device->handle()->newLibrary(source, options, &error);
            source->release();
            options->release();
            if (error != nullptr) {
                LUISA_WARNING_WITH_LOCATION(
                    "Error during compiling Metal shader '{}': {}.",
                    name, error->localizedDescription()->utf8String());
            }
        }
        LUISA_ASSERT(library != nullptr, "Failed to compile Metal shader '{}'.", name);

        // create pso
        auto compute_pipeline_desc = MTL::ComputePipelineDescriptor::alloc()->init();
        compute_pipeline_desc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
        compute_pipeline_desc->setMaxTotalThreadsPerThreadgroup(block_size.x * block_size.y * block_size.z);
        auto function = library->newFunction(MTLSTR("kernel_main"));
        compute_pipeline_desc->setComputeFunction(function);
        NS::Error *error = nullptr;
        auto pipeline = _device->handle()->newComputePipelineState(
            compute_pipeline_desc, MTL::PipelineOptionNone, nullptr, &error);
        if (error != nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "Error during creating Metal compute pipeline for '{}': {}.",
                name, error->localizedDescription()->utf8String());
        }
        LUISA_ASSERT(pipeline != nullptr, "Failed to create Metal compute pipeline for '{}'.", name);

        // store the library
        if (uses_cache) {
            _store_disk_archive(hash, name, option, block_size, compute_pipeline_desc);
        }

        function->release();
        compute_pipeline_desc->release();

        library->release();

        auto shared_pipeline = NS::TransferPtr(pipeline);
        _cache.update(hash, shared_pipeline);
        return shared_pipeline;
    });
}

}// namespace luisa::compute::metal
